# Jetson GStreamer 360° AVM 拼接插件

`nvavmstitch` 是一个面向 NVIDIA Jetson 的四路环视 GStreamer 插件。插件
使用 CUDA 直接处理 NVMM 中的 NV12 图像，根据预计算的 AVMAP v1 标定资产，
完成鱼眼校正、鸟瞰变换、融合权重和车身覆盖图合成。

当前版本主要适配 Jetson Orin NX、JetPack 5.1.2、L4T 35.4.1 和 CUDA 11.4。
四路输入均为 1920×1080 NV12/NVMM，固定输入 Pad 为：

- `sink_front`：前摄像头
- `sink_left`：左摄像头
- `sink_right`：右摄像头
- `sink_bottom`：后摄像头（历史命名保留为 bottom）

## 标定资产

仓库中的 `calibration/标定文件.zip` 包含：

```text
avm_zhuangzaiji2_quick.bin
```

首次部署时解压：

```bash
cd calibration
sha256sum -c SHA256SUMS
unzip 标定文件.zip
sudo install -D -m 0644 avm_zhuangzaiji2_quick.bin \
  /opt/calibration/avm_zhuangzaiji2_quick.bin
```

## 依赖检查

天准域控通常已经预装所需组件。先检查，不要直接重复安装：

```bash
packages=(
  build-essential
  cmake
  pkg-config
  libgstreamer1.0-dev
  libgstreamer-plugins-base1.0-dev
  nvidia-l4t-jetson-multimedia-api
)

missing=()
for package in "${packages[@]}"; do
  if dpkg-query -W -f='${Status}' "$package" 2>/dev/null | \
      grep -q '^install ok installed$'; then
    echo "[已安装] $package"
  else
    echo "[缺少]   $package"
    missing+=("$package")
  fi
done

if ((${#missing[@]} == 0)); then
  echo "依赖检查通过，不需要安装。"
else
  echo "缺少的软件包：${missing[*]}"
fi
```

只有设备可以访问对应软件源时，才执行：

```bash
if ((${#missing[@]} > 0)); then
  sudo apt-get install -y "${missing[@]}"
fi
```

离线域控不要执行 `apt upgrade`，避免改变 JetPack/L4T 组件版本。

## 编译和安装

```bash
export PATH=/usr/local/cuda/bin:$PATH

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGSTREAMER_PLUGIN_DIR=/usr/lib/aarch64-linux-gnu/gstreamer-1.0

cmake --build build --parallel
sudo cmake --install build
gst-inspect-1.0 nvavmstitch
```

## 使用要点

创建元素时指定标定资产：

```text
nvavmstitch name=stitch \
  asset-file=/opt/calibration/avm_zhuangzaiji2_quick.bin \
  output-width=800 output-height=900 fit-mode=contain
```

四个摄像头分支分别连接：

```text
... ! nvvidconv bl-output=false ! queue ! stitch.sink_front
... ! nvvidconv bl-output=false ! queue ! stitch.sink_left
... ! nvvidconv bl-output=false ! queue ! stitch.sink_right
... ! nvvidconv bl-output=false ! queue ! stitch.sink_bottom
```

每个分支都必须输出 1920×1080 NV12/NVMM，并建议使用
`nvvidconv bl-output=false` 得到 pitch-linear 表面。插件输出可以继续连接
`nvv4l2h265enc` 编码、RTP/UDP 推流或本地显示。

## 调试

```bash
GST_DEBUG=nvavmstitch:6 gst-launch-1.0 ...
sudo tegrastats
```

不要让测试管线和现有推流服务同时打开同一个 `/dev/video*` 设备。
