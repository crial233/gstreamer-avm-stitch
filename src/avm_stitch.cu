#include "avm_stitch.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace {
__device__ uint8_t sample_u8(const uint8_t *p, size_t pitch, int width, int height, float x, float y) {
  if (x < 0.f || y < 0.f || x > width - 1.f || y > height - 1.f) return 0;
  int x0 = min(max(static_cast<int>(floorf(x)), 0), width - 1);
  int y0 = min(max(static_cast<int>(floorf(y)), 0), height - 1);
  int x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1);
  float fx = x - x0, fy = y - y0;
  const uint8_t *r0 = p + static_cast<size_t>(y0) * pitch;
  const uint8_t *r1 = p + static_cast<size_t>(y1) * pitch;
  float v = (1-fy)*((1-fx)*r0[x0]+fx*r0[x1]) + fy*((1-fx)*r1[x0]+fx*r1[x1]);
  return static_cast<uint8_t>(fminf(fmaxf(v + .5f, 0.f), 255.f));
}

__device__ uchar2 sample_uv(const uint8_t *p, size_t pitch, int width, int height, float x, float y) {
  const int cw = width / 2, ch = height / 2;
  float cx = x * .5f, cy = y * .5f;
  if (cx < 0.f || cy < 0.f || cx > cw - 1.f || cy > ch - 1.f) return make_uchar2(128,128);
  int x0=min(max((int)floorf(cx),0),cw-1), y0=min(max((int)floorf(cy),0),ch-1);
  int x1=min(x0+1,cw-1), y1=min(y0+1,ch-1); float fx=cx-x0, fy=cy-y0;
  const uchar2 *r0=reinterpret_cast<const uchar2 *>(p+(size_t)y0*pitch);
  const uchar2 *r1=reinterpret_cast<const uchar2 *>(p+(size_t)y1*pitch);
  float u=(1-fy)*((1-fx)*r0[x0].x+fx*r0[x1].x)+fy*((1-fx)*r1[x0].x+fx*r1[x1].x);
  float v=(1-fy)*((1-fx)*r0[x0].y+fx*r0[x1].y)+fy*((1-fx)*r1[x0].y+fx*r1[x1].y);
  return make_uchar2((uint8_t)(u+.5f),(uint8_t)(v+.5f));
}

__device__ bool output_to_canvas(int ox, int oy, int ow, int oh, int cw, int ch, int mode,
                                 float &cx, float &cy) {
  if (mode == 2) { cx=(ox+.5f)*cw/ow-.5f; cy=(oy+.5f)*ch/oh-.5f; return true; }
  float sx=(float)ow/cw, sy=(float)oh/ch;
  float scale = mode == 0 ? fminf(sx,sy) : fmaxf(sx,sy);
  float dw=cw*scale, dh=ch*scale;
  cx=(ox-(ow-dw)*.5f+.5f)/scale-.5f; cy=(oy-(oh-dh)*.5f+.5f)/scale-.5f;
  return cx>=0.f && cy>=0.f && cx<cw && cy<ch;
}

__global__ void stitch_y_kernel(uint8_t *out, size_t pitch, int ow, int oh, int sw, int sh,
                                int cw, int ch, int mode, const float *mx, const float *my,
                                const float *wt, const uint8_t *overlay,
                                const uint8_t *y0,size_t p0,const uint8_t *y1,size_t p1,
                                const uint8_t *y2,size_t p2,const uint8_t *y3,size_t p3) {
  int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=ow||y>=oh)return; float cx,cy; uint8_t value=16;
  if(output_to_canvas(x,y,ow,oh,cw,ch,mode,cx,cy)) {
    int ix=min(max((int)lrintf(cx),0),cw-1), iy=min(max((int)lrintf(cy),0),ch-1);
    int pi=iy*cw+ix, n=cw*ch; const uint8_t *ys[4]={y0,y1,y2,y3}; size_t ps[4]={p0,p1,p2,p3};
    float sum=0.f, wsum=0.f;
    #pragma unroll
    for(int c=0;c<4;c++){float w=wt[c*n+pi]; if(w>0.f){sum+=w*sample_u8(ys[c],ps[c],sw,sh,mx[c*n+pi],my[c*n+pi]);wsum+=w;}}
    if(wsum>1e-6f)value=(uint8_t)fminf(fmaxf(sum/wsum+.5f,0.f),255.f);
    uint8_t a=overlay[pi*4+3]; if(a)value=(uint8_t)((value*(255-a)+overlay[pi*4]*a+127)/255);
  }
  out[(size_t)y*pitch+x]=value;
}

__global__ void stitch_uv_kernel(uint8_t *out,size_t pitch,int ow,int oh,int sw,int sh,int cw,int ch,
                                 int mode,const float *mx,const float *my,const float *wt,const uint8_t *overlay,
                                 const uint8_t *u0,size_t p0,const uint8_t *u1,size_t p1,
                                 const uint8_t *u2,size_t p2,const uint8_t *u3,size_t p3) {
  int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=ow/2||y>=oh/2)return; float cx,cy; uchar2 value=make_uchar2(128,128);
  if(output_to_canvas(x*2,y*2,ow,oh,cw,ch,mode,cx,cy)){
    int ix=min(max((int)lrintf(cx),0),cw-1),iy=min(max((int)lrintf(cy),0),ch-1),pi=iy*cw+ix,n=cw*ch;
    const uint8_t *us[4]={u0,u1,u2,u3};size_t ps[4]={p0,p1,p2,p3};float su=0,sv=0,ws=0;
    #pragma unroll
    for(int c=0;c<4;c++){float w=wt[c*n+pi];if(w>0){uchar2 q=sample_uv(us[c],ps[c],sw,sh,mx[c*n+pi],my[c*n+pi]);su+=w*q.x;sv+=w*q.y;ws+=w;}}
    if(ws>1e-6f)value=make_uchar2((uint8_t)(su/ws+.5f),(uint8_t)(sv/ws+.5f));
    uint8_t a=overlay[pi*4+3];if(a){value.x=(value.x*(255-a)+overlay[pi*4+1]*a+127)/255;value.y=(value.y*(255-a)+overlay[pi*4+2]*a+127)/255;}
  }
  reinterpret_cast<uchar2 *>(out+(size_t)y*pitch)[x]=value;
}

bool cuda_ok(cudaError_t r,const char *what,std::string &error){if(r==cudaSuccess)return true;error=std::string(what)+": "+cudaGetErrorString(r);return false;}
bool valid_frame(CUeglFrame &f,uint32_t w,uint32_t h,std::string &e){
  if(f.planeCount<2||f.width!=w||f.height!=h){e="CUDA EGL frame dimensions/planes mismatch";return false;}
  if(f.frameType!=CU_EGL_FRAME_TYPE_ARRAY&&f.frameType!=CU_EGL_FRAME_TYPE_PITCH){e="unsupported CUDA EGL frame type";return false;}
  return true;
}
} // namespace

AvmStitcher::~AvmStitcher(){reset();}
bool AvmStitcher::initialize(const AvmAsset&a,uint32_t ow,uint32_t oh,AvmFitMode mode,std::string&e){
  reset();source_width_=a.header.source_width;source_height_=a.header.source_height;canvas_width_=a.header.canvas_width;canvas_height_=a.header.canvas_height;
  output_width_=ow?ow:canvas_width_;output_height_=oh?oh:canvas_height_;fit_mode_=mode;
  if((output_width_&1)||(output_height_&1)){e="NV12 output dimensions must be even";return false;}
  size_t cp=a.map_x.size()*sizeof(float),op=a.overlay_yuva.size();
  if(!cuda_ok(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking),"cudaStreamCreate",e)||
     !cuda_ok(cudaMalloc(reinterpret_cast<void **>(&map_x_),cp),"cudaMalloc map_x",e)||!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&map_y_),cp),"cudaMalloc map_y",e)||
     !cuda_ok(cudaMalloc(reinterpret_cast<void **>(&weights_),cp),"cudaMalloc weights",e)||!cuda_ok(cudaMalloc(reinterpret_cast<void **>(&overlay_),op),"cudaMalloc overlay",e))return false;
  cudaMemcpyAsync(map_x_,a.map_x.data(),cp,cudaMemcpyHostToDevice,stream_);cudaMemcpyAsync(map_y_,a.map_y.data(),cp,cudaMemcpyHostToDevice,stream_);
  cudaMemcpyAsync(weights_,a.weights.data(),cp,cudaMemcpyHostToDevice,stream_);cudaMemcpyAsync(overlay_,a.overlay_yuva.data(),op,cudaMemcpyHostToDevice,stream_);
  for(int i=0;i<4;i++){if(!cuda_ok(cudaMallocPitch(reinterpret_cast<void **>(&input_y_[i]),&input_y_pitch_[i],source_width_,source_height_),"cudaMallocPitch input Y",e)||
    !cuda_ok(cudaMallocPitch(reinterpret_cast<void **>(&input_uv_[i]),&input_uv_pitch_[i],source_width_,source_height_/2),"cudaMallocPitch input UV",e))return false;}
  if(!cuda_ok(cudaMallocPitch(reinterpret_cast<void **>(&output_y_),&output_y_pitch_,output_width_,output_height_),"cudaMallocPitch output Y",e)||
     !cuda_ok(cudaMallocPitch(reinterpret_cast<void **>(&output_uv_),&output_uv_pitch_,output_width_,output_height_/2),"cudaMallocPitch output UV",e))return false;
  return cuda_ok(cudaStreamSynchronize(stream_),"asset upload",e);
}
void AvmStitcher::reset(){for(int i=0;i<4;i++){cudaFree(input_y_[i]);cudaFree(input_uv_[i]);input_y_[i]=input_uv_[i]=nullptr;}cudaFree(output_y_);cudaFree(output_uv_);cudaFree(map_x_);cudaFree(map_y_);cudaFree(weights_);cudaFree(overlay_);output_y_=output_uv_=nullptr;map_x_=map_y_=weights_=nullptr;overlay_=nullptr;if(stream_)cudaStreamDestroy(stream_);stream_=nullptr;}
bool AvmStitcher::process(CUeglFrame in[4],CUeglFrame&out,std::string&e){
  for(int i=0;i<4;i++)if(!valid_frame(in[i],source_width_,source_height_,e))return false;if(!valid_frame(out,output_width_,output_height_,e))return false;
  for(int i=0;i<4;i++){
    cudaError_t r;
    if(in[i].frameType==CU_EGL_FRAME_TYPE_ARRAY){
      r=cudaMemcpy2DFromArrayAsync(input_y_[i],input_y_pitch_[i],reinterpret_cast<cudaArray_t>(in[i].frame.pArray[0]),0,0,source_width_,source_height_,cudaMemcpyDeviceToDevice,stream_);
      if(r==cudaSuccess)r=cudaMemcpy2DFromArrayAsync(input_uv_[i],input_uv_pitch_[i],reinterpret_cast<cudaArray_t>(in[i].frame.pArray[1]),0,0,source_width_,source_height_/2,cudaMemcpyDeviceToDevice,stream_);
    }else{
      r=cudaMemcpy2DAsync(input_y_[i],input_y_pitch_[i],in[i].frame.pPitch[0],in[i].pitch,source_width_,source_height_,cudaMemcpyDeviceToDevice,stream_);
      if(r==cudaSuccess)r=cudaMemcpy2DAsync(input_uv_[i],input_uv_pitch_[i],in[i].frame.pPitch[1],in[i].pitch,source_width_,source_height_/2,cudaMemcpyDeviceToDevice,stream_);
    }
    if(!cuda_ok(r,"copy AVM input",e))return false;
  }
  dim3 b(16,16),gy((output_width_+15)/16,(output_height_+15)/16),gu((output_width_/2+15)/16,(output_height_/2+15)/16);
  stitch_y_kernel<<<gy,b,0,stream_>>>(output_y_,output_y_pitch_,output_width_,output_height_,source_width_,source_height_,canvas_width_,canvas_height_,(int)fit_mode_,map_x_,map_y_,weights_,overlay_,input_y_[0],input_y_pitch_[0],input_y_[1],input_y_pitch_[1],input_y_[2],input_y_pitch_[2],input_y_[3],input_y_pitch_[3]);
  stitch_uv_kernel<<<gu,b,0,stream_>>>(output_uv_,output_uv_pitch_,output_width_,output_height_,source_width_,source_height_,canvas_width_,canvas_height_,(int)fit_mode_,map_x_,map_y_,weights_,overlay_,input_uv_[0],input_uv_pitch_[0],input_uv_[1],input_uv_pitch_[1],input_uv_[2],input_uv_pitch_[2],input_uv_[3],input_uv_pitch_[3]);
  if(!cuda_ok(cudaGetLastError(),"AVM kernel launch",e))return false;
  cudaError_t r;
  if(out.frameType==CU_EGL_FRAME_TYPE_ARRAY){
    r=cudaMemcpy2DToArrayAsync(reinterpret_cast<cudaArray_t>(out.frame.pArray[0]),0,0,output_y_,output_y_pitch_,output_width_,output_height_,cudaMemcpyDeviceToDevice,stream_);
    if(r==cudaSuccess)r=cudaMemcpy2DToArrayAsync(reinterpret_cast<cudaArray_t>(out.frame.pArray[1]),0,0,output_uv_,output_uv_pitch_,output_width_,output_height_/2,cudaMemcpyDeviceToDevice,stream_);
  }else{
    r=cudaMemcpy2DAsync(out.frame.pPitch[0],out.pitch,output_y_,output_y_pitch_,output_width_,output_height_,cudaMemcpyDeviceToDevice,stream_);
    if(r==cudaSuccess)r=cudaMemcpy2DAsync(out.frame.pPitch[1],out.pitch,output_uv_,output_uv_pitch_,output_width_,output_height_/2,cudaMemcpyDeviceToDevice,stream_);
  }
  if(!cuda_ok(r,"copy AVM output",e))return false;
  return cuda_ok(cudaStreamSynchronize(stream_),"AVM processing",e);
}
