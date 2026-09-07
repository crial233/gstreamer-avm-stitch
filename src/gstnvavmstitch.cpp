#include "avm_asset.hpp"
#include "avm_stitch.hpp"

#include <gst/base/gstaggregator.h>
#include <gst/video/video.h>
#include <cudaEGL.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvbufsurface.h>

#include <array>
#include <memory>
#include <string>

GST_DEBUG_CATEGORY_STATIC(gst_nv_avm_debug);
#define GST_CAT_DEFAULT gst_nv_avm_debug

typedef struct _GstNvAvmStitch {
  GstAggregator parent;
  gchar *asset_file;
  guint output_width;
  guint output_height;
  gint fit_mode;
  guint device_id;
  AvmAsset *asset;
  AvmStitcher *stitcher;
  guint64 frame_count;
  gboolean time_segment_sent;
} GstNvAvmStitch;

typedef struct _GstNvAvmStitchClass { GstAggregatorClass parent_class; } GstNvAvmStitchClass;

#define GST_TYPE_NV_AVM_STITCH (gst_nv_avm_stitch_get_type())
#define GST_NV_AVM_STITCH(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_NV_AVM_STITCH, GstNvAvmStitch))
G_DEFINE_TYPE(GstNvAvmStitch, gst_nv_avm_stitch, GST_TYPE_AGGREGATOR)

enum { PROP_0, PROP_ASSET_FILE, PROP_OUTPUT_WIDTH, PROP_OUTPUT_HEIGHT, PROP_FIT_MODE, PROP_DEVICE_ID };

static GstStaticPadTemplate front_template = GST_STATIC_PAD_TEMPLATE("sink_front", GST_PAD_SINK, GST_PAD_ALWAYS,
  GST_STATIC_CAPS("video/x-raw(memory:NVMM),format=(string)NV12,width=(int)[2,MAX],height=(int)[2,MAX],framerate=(fraction)[0/1,MAX]"));
static GstStaticPadTemplate left_template = GST_STATIC_PAD_TEMPLATE("sink_left", GST_PAD_SINK, GST_PAD_ALWAYS,
  GST_STATIC_CAPS("video/x-raw(memory:NVMM),format=(string)NV12,width=(int)[2,MAX],height=(int)[2,MAX],framerate=(fraction)[0/1,MAX]"));
static GstStaticPadTemplate right_template = GST_STATIC_PAD_TEMPLATE("sink_right", GST_PAD_SINK, GST_PAD_ALWAYS,
  GST_STATIC_CAPS("video/x-raw(memory:NVMM),format=(string)NV12,width=(int)[2,MAX],height=(int)[2,MAX],framerate=(fraction)[0/1,MAX]"));
static GstStaticPadTemplate bottom_template = GST_STATIC_PAD_TEMPLATE("sink_bottom", GST_PAD_SINK, GST_PAD_ALWAYS,
  GST_STATIC_CAPS("video/x-raw(memory:NVMM),format=(string)NV12,width=(int)[2,MAX],height=(int)[2,MAX],framerate=(fraction)[0/1,MAX]"));
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS,
  GST_STATIC_CAPS("video/x-raw(memory:NVMM),format=(string)NV12,width=(int)[2,MAX],height=(int)[2,MAX],framerate=(fraction)[0/1,MAX]"));

struct MappedSurface {
  GstBuffer *buffer = nullptr;
  GstMapInfo map = GST_MAP_INFO_INIT;
  NvBufSurface *surface = nullptr;
  CUgraphicsResource resource = nullptr;
  CUeglFrame frame{};
  bool gst_mapped = false, egl_mapped = false;
  bool open(GstBuffer *b, guint expected_w, guint expected_h, std::string &error) {
    buffer = b;
    if (!gst_buffer_map(b, &map, GST_MAP_READ)) { error="gst_buffer_map failed"; return false; }
    gst_mapped=true; surface=reinterpret_cast<NvBufSurface *>(map.data);
    if (!surface || surface->batchSize<1) { error="invalid NvBufSurface"; return false; }
    auto &p=surface->surfaceList[0];
    if (p.width!=expected_w || p.height!=expected_h || p.planeParams.num_planes<2) { error="NvBufSurface dimensions/planes mismatch"; return false; }
    if (NvBufSurfaceMapEglImage(surface,0)!=0) { error="NvBufSurfaceMapEglImage failed"; return false; }
    egl_mapped=true; auto image=static_cast<EGLImageKHR>(p.mappedAddr.eglImage);
    CUresult r=cuGraphicsEGLRegisterImage(&resource,image,CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if(r!=CUDA_SUCCESS){const char*s=nullptr;cuGetErrorString(r,&s);error=std::string("cuGraphicsEGLRegisterImage: ")+(s?s:"unknown");return false;}
    r=cuGraphicsResourceGetMappedEglFrame(&frame,resource,0,0);
    if(r!=CUDA_SUCCESS){const char*s=nullptr;cuGetErrorString(r,&s);error=std::string("cuGraphicsResourceGetMappedEglFrame: ")+(s?s:"unknown");return false;}
    return true;
  }
  void close(){if(resource)cuGraphicsUnregisterResource(resource);resource=nullptr;if(egl_mapped)NvBufSurfaceUnMapEglImage(surface,0);egl_mapped=false;if(gst_mapped)gst_buffer_unmap(buffer,&map);gst_mapped=false;surface=nullptr;buffer=nullptr;}
  ~MappedSurface(){close();}
};

static void destroy_nvbuf(gpointer data) { if (data) NvBufSurfaceDestroy(static_cast<NvBufSurface *>(data)); }

static GstBuffer *allocate_nvmm(guint width, guint height, guint device_id, std::string &error) {
  NvBufSurfaceCreateParams params{}; params.gpuId=device_id; params.width=width; params.height=height;
  params.colorFormat=NVBUF_COLOR_FORMAT_NV12; params.layout=NVBUF_LAYOUT_PITCH; params.memType=NVBUF_MEM_SURFACE_ARRAY;
  NvBufSurface *surface=nullptr;
  if(NvBufSurfaceCreate(&surface,1,&params)!=0 || !surface){error="NvBufSurfaceCreate failed";return nullptr;}
  GstBuffer *buffer=gst_buffer_new_wrapped_full(GST_MEMORY_FLAG_PHYSICALLY_CONTIGUOUS,
    surface,sizeof(NvBufSurface),0,sizeof(NvBufSurface),surface,destroy_nvbuf);
  if(!buffer){NvBufSurfaceDestroy(surface);error="gst_buffer_new_wrapped_full failed";}
  return buffer;
}

static void set_property(GObject *o,guint id,const GValue *v,GParamSpec *p){auto*self=GST_NV_AVM_STITCH(o);switch(id){
 case PROP_ASSET_FILE:g_free(self->asset_file);self->asset_file=g_value_dup_string(v);break;
 case PROP_OUTPUT_WIDTH:self->output_width=g_value_get_uint(v);break;case PROP_OUTPUT_HEIGHT:self->output_height=g_value_get_uint(v);break;
 case PROP_FIT_MODE:self->fit_mode=g_value_get_enum(v);break;case PROP_DEVICE_ID:self->device_id=g_value_get_uint(v);break;
 default:G_OBJECT_WARN_INVALID_PROPERTY_ID(o,id,p);}}
static void get_property(GObject *o,guint id,GValue *v,GParamSpec *p){auto*self=GST_NV_AVM_STITCH(o);switch(id){
 case PROP_ASSET_FILE:g_value_set_string(v,self->asset_file);break;case PROP_OUTPUT_WIDTH:g_value_set_uint(v,self->output_width);break;
 case PROP_OUTPUT_HEIGHT:g_value_set_uint(v,self->output_height);break;case PROP_FIT_MODE:g_value_set_enum(v,self->fit_mode);break;
 case PROP_DEVICE_ID:g_value_set_uint(v,self->device_id);break;default:G_OBJECT_WARN_INVALID_PROPERTY_ID(o,id,p);}}

static gboolean start(GstAggregator *agg){auto*self=GST_NV_AVM_STITCH(agg);self->frame_count=0;self->time_segment_sent=FALSE;if(!self->asset_file||!*self->asset_file){GST_ELEMENT_ERROR(self,RESOURCE,NOT_FOUND,("asset-file is required"),(nullptr));return FALSE;}
 std::string e;if(!load_avm_asset(self->asset_file,*self->asset,e)){GST_ELEMENT_ERROR(self,RESOURCE,READ,("Failed to load AVM asset"),("%s",e.c_str()));return FALSE;}
 guint w=self->output_width?self->output_width:self->asset->header.canvas_width,h=self->output_height?self->output_height:self->asset->header.canvas_height;
 if(!self->stitcher->initialize(*self->asset,w,h,static_cast<AvmFitMode>(self->fit_mode),e)){GST_ELEMENT_ERROR(self,LIBRARY,INIT,("Failed to initialize AVM CUDA core"),("%s",e.c_str()));return FALSE;}
 GstCaps*caps=gst_caps_new_simple("video/x-raw","format",G_TYPE_STRING,"NV12","width",G_TYPE_INT,w,"height",G_TYPE_INT,h,"framerate",GST_TYPE_FRACTION,30,1,nullptr);
 GstCapsFeatures*f=gst_caps_features_new("memory:NVMM",nullptr);gst_caps_set_features(caps,0,f);gst_aggregator_set_src_caps(agg,caps);gst_caps_unref(caps);
 GST_INFO_OBJECT(self,"AVM ready: source=%ux%u canvas=%ux%u output=%ux%u",self->asset->header.source_width,self->asset->header.source_height,self->asset->header.canvas_width,self->asset->header.canvas_height,w,h);return TRUE;}
static gboolean stop(GstAggregator *agg){auto*self=GST_NV_AVM_STITCH(agg);self->stitcher->reset();*self->asset=AvmAsset{};return TRUE;}

static GstFlowReturn aggregate(GstAggregator *agg,gboolean timeout){(void)timeout;auto*self=GST_NV_AVM_STITCH(agg);
 if(self->frame_count<5||self->frame_count%30==0)GST_INFO_OBJECT(self,"aggregate begin frame=%" G_GUINT64_FORMAT " timeout=%d",self->frame_count,timeout);
 const char*names[4]={"sink_front","sink_left","sink_right","sink_bottom"};std::array<GstAggregatorPad*,4> pads{};std::array<GstBuffer*,4> bufs{};
 for(int i=0;i<4;i++){GstPad*p=gst_element_get_static_pad(GST_ELEMENT(self),names[i]);pads[i]=GST_AGGREGATOR_PAD(p);bufs[i]=gst_aggregator_pad_peek_buffer(pads[i]);if(!bufs[i]){gst_object_unref(p);for(int j=0;j<i;j++){gst_buffer_unref(bufs[j]);gst_object_unref(pads[j]);}return GST_FLOW_OK;}}
 guint ow=self->output_width?self->output_width:self->asset->header.canvas_width,oh=self->output_height?self->output_height:self->asset->header.canvas_height;std::string e;
 GstBuffer*out=allocate_nvmm(ow,oh,self->device_id,e);if(!out){GST_ELEMENT_ERROR(self,RESOURCE,NO_SPACE_LEFT,("Failed to allocate AVM output"),("%s",e.c_str()));for(auto*b:bufs)gst_buffer_unref(b);for(auto*p:pads)gst_object_unref(p);return GST_FLOW_ERROR;}
 cudaSetDevice(self->device_id);std::array<MappedSurface,4> input;MappedSurface output;bool ok=true;
 for(int i=0;i<4&&ok;i++)ok=input[i].open(bufs[i],self->asset->header.source_width,self->asset->header.source_height,e);
 if(ok)ok=output.open(out,ow,oh,e);CUeglFrame frames[4]{};if(ok){for(int i=0;i<4;i++)frames[i]=input[i].frame;ok=self->stitcher->process(frames,output.frame,e);}output.close();for(auto&i:input)i.close();
 if(self->frame_count<5||self->frame_count%30==0)GST_INFO_OBJECT(self,"aggregate processed frame=%" G_GUINT64_FORMAT " ok=%d",self->frame_count,ok);
 for(auto*b:bufs)gst_buffer_unref(b);for(auto*p:pads){GstBuffer*b=gst_aggregator_pad_pop_buffer(p);if(b)gst_buffer_unref(b);gst_object_unref(p);}if(!ok){gst_buffer_unref(out);GST_ELEMENT_ERROR(self,STREAM,FAILED,("AVM processing failed"),("%s",e.c_str()));return GST_FLOW_ERROR;}
 if(!self->time_segment_sent){GstSegment segment;gst_segment_init(&segment,GST_FORMAT_TIME);segment.start=0;segment.time=0;segment.position=0;
  if(!gst_pad_push_event(GST_AGGREGATOR_SRC_PAD(agg),gst_event_new_segment(&segment))){gst_buffer_unref(out);GST_ELEMENT_ERROR(self,STREAM,FAILED,("Failed to send AVM TIME segment"),(nullptr));return GST_FLOW_ERROR;}self->time_segment_sent=TRUE;}
 GstClockTime pts=gst_util_uint64_scale(self->frame_count,GST_SECOND,30),next=gst_util_uint64_scale(self->frame_count+1,GST_SECOND,30);
 GST_BUFFER_PTS(out)=pts;GST_BUFFER_DTS(out)=GST_CLOCK_TIME_NONE;GST_BUFFER_DURATION(out)=next-pts;GST_BUFFER_OFFSET(out)=self->frame_count;GST_BUFFER_OFFSET_END(out)=self->frame_count+1;
 if(self->frame_count==0)GST_BUFFER_FLAG_SET(out,GST_BUFFER_FLAG_DISCONT);self->frame_count++;return gst_aggregator_finish_buffer(agg,out);}

static void finalize(GObject*o){auto*self=GST_NV_AVM_STITCH(o);delete self->stitcher;delete self->asset;g_free(self->asset_file);G_OBJECT_CLASS(gst_nv_avm_stitch_parent_class)->finalize(o);}
static gboolean sink_event(GstAggregator*agg,GstAggregatorPad*pad,GstEvent*event){
 if(GST_EVENT_TYPE(event)==GST_EVENT_SEGMENT){const GstSegment*incoming=nullptr;gst_event_parse_segment(event,&incoming);if(incoming&&incoming->format!=GST_FORMAT_TIME){
   GstSegment normalized;gst_segment_init(&normalized,GST_FORMAT_TIME);normalized.start=0;normalized.time=0;normalized.position=0;gst_event_unref(event);event=gst_event_new_segment(&normalized);
 }}
 return GST_AGGREGATOR_CLASS(gst_nv_avm_stitch_parent_class)->sink_event(agg,pad,event);
}
static GType fit_mode_type(){static GType t=0;if(!t){static const GEnumValue v[]={{0,"Keep aspect ratio and letterbox","contain"},{1,"Keep aspect ratio and crop","cover"},{2,"Resize independently","stretch"},{0,nullptr,nullptr}};t=g_enum_register_static("GstNvAvmFitMode",v);}return t;}
static void gst_nv_avm_stitch_class_init(GstNvAvmStitchClass*k){auto*g=G_OBJECT_CLASS(k);auto*e=GST_ELEMENT_CLASS(k);auto*a=GST_AGGREGATOR_CLASS(k);g->set_property=set_property;g->get_property=get_property;g->finalize=finalize;
 g_object_class_install_property(g,PROP_ASSET_FILE,g_param_spec_string("asset-file","AVM asset","AVMAP v1 calibration asset",nullptr,(GParamFlags)(G_PARAM_READWRITE|G_PARAM_STATIC_STRINGS)));
 g_object_class_install_property(g,PROP_OUTPUT_WIDTH,g_param_spec_uint("output-width","Output width","0 uses native canvas",0,8192,0,(GParamFlags)(G_PARAM_READWRITE|G_PARAM_STATIC_STRINGS)));
 g_object_class_install_property(g,PROP_OUTPUT_HEIGHT,g_param_spec_uint("output-height","Output height","0 uses native canvas",0,8192,0,(GParamFlags)(G_PARAM_READWRITE|G_PARAM_STATIC_STRINGS)));
 g_object_class_install_property(g,PROP_FIT_MODE,g_param_spec_enum("fit-mode","Fit mode","contain, cover or stretch",fit_mode_type(),0,(GParamFlags)(G_PARAM_READWRITE|G_PARAM_STATIC_STRINGS)));
 g_object_class_install_property(g,PROP_DEVICE_ID,g_param_spec_uint("device-id","CUDA device","CUDA device index",0,16,0,(GParamFlags)(G_PARAM_READWRITE|G_PARAM_STATIC_STRINGS)));
 gst_element_class_add_static_pad_template(e,&front_template);gst_element_class_add_static_pad_template(e,&left_template);gst_element_class_add_static_pad_template(e,&right_template);gst_element_class_add_static_pad_template(e,&bottom_template);gst_element_class_add_static_pad_template(e,&src_template);
 gst_element_class_set_static_metadata(e,"Jetson four-camera AVM stitcher","Filter/Editor/Video","CUDA/NVMM 360 surround-view stitcher","OpenAI");a->start=GST_DEBUG_FUNCPTR(start);a->stop=GST_DEBUG_FUNCPTR(stop);a->sink_event=GST_DEBUG_FUNCPTR(sink_event);a->aggregate=GST_DEBUG_FUNCPTR(aggregate);}
static void add_aggregator_pad(GstNvAvmStitch *self, GstStaticPadTemplate *static_template,
                               const char *name) {
  GstPadTemplate *pad_template = gst_static_pad_template_get(static_template);
  GstPad *pad = GST_PAD(g_object_new(GST_TYPE_AGGREGATOR_PAD,
                                    "name", name,
                                    "direction", GST_PAD_SINK,
                                    "template", pad_template,
                                    nullptr));
  gst_object_unref(pad_template);
  if (!gst_element_add_pad(GST_ELEMENT(self), pad)) {
    GST_ERROR_OBJECT(self, "failed to add aggregator pad %s", name);
    gst_object_unref(pad);
  }
}

static void gst_nv_avm_stitch_init(GstNvAvmStitch*self){
  self->asset_file=nullptr;self->output_width=self->output_height=0;self->fit_mode=0;self->device_id=0;
  self->asset=new AvmAsset();self->stitcher=new AvmStitcher();self->frame_count=0;self->time_segment_sent=FALSE;
  add_aggregator_pad(self, &front_template, "sink_front");
  add_aggregator_pad(self, &left_template, "sink_left");
  add_aggregator_pad(self, &right_template, "sink_right");
  add_aggregator_pad(self, &bottom_template, "sink_bottom");
}

static gboolean plugin_init(GstPlugin *plugin) {
  GST_DEBUG_CATEGORY_INIT(gst_nv_avm_debug, "nvavmstitch", 0,
                          "Jetson four-camera AVM stitcher");
  return gst_element_register(plugin, "nvavmstitch", GST_RANK_NONE,
                              GST_TYPE_NV_AVM_STITCH);
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, nvavmstitch,
                  "Jetson CUDA/NVMM four-camera AVM stitcher", plugin_init,
                  "0.1.0", "LGPL", "gstnvavmstitch", "https://openai.com")
