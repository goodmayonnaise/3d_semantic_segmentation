* ResNeSt
* TransU-Net 
* Inception module
* MSCA(Multi-scale Channel Attention)
* SE block 
* VGG block
* ASPP 
* SalsaNeXt
* DoubleU-Net 
* DoubleU-ResNeSt
* SC-UNet
  - LiDAR point cloud data to RGB
* af2-s3 AF2M, AFSM 모듈 2D로 구현 (open source 없음)
  - ref : https://arxiv.org/pdf/2102.04530v1.pdf
* ASPP 
  - 마지막 conv layer stride=2 변경, down sampling으로 사용 
-----------------------------------------------------------------
* DataLoaders/semantic_kitti.py
  - 전처리 없이 바로 불러오기 가능
  - 전방 영역 기능 추가
  - pcd projection 후 pad=1 기능 추가 
