# opencv-practice

## 이 repository는 opencv라이브러리를 사용하여 여러가지 프로젝트를 진행한 코드를 모아 둔 레포지토리이다. 

### gaussian.cpp 파일
- 가우시안 필터의 사이즈를 변경시키면서 히스토그램을 분석해보는 코드이다.

### filter1.cpp 파일
- fourier transform을 이용하여 영상을 주파수 영역으로 변환한 뒤 주파수 영역 필터링을 통해 영상에 어떤 영향을 미치는 지 알아볼 수 있는 코드이다.

### 6week.cpp 파일 
- median filter를 직접 구현한 뒤 salt-pepper noise가 있는 영상에 적용하여 noise를 없애보는 코드
- canny-edge-detector를 이용해서 영상의 edge를 찾아내보는 코드
- bilateral filter를 구현하여 영상에 적용시켜 보는 코드

### color.cpp파일
- 영상의 color표현 방식을 바꿔보는 코드
- opencv라이브러리를 사용하면 BGR2HSV와 같은 함수를 사용하여 바로 가능하나 low-level로 직접 구현하였다. 
- k-means clustering을 low-level로 직접 구현한 코드 
- k-means clustering을 통해 군집화 시킨 뒤 랜덤으로 영상의 색을 변환시켜 보는 코드

### implementation.cpp파일
- 세 가지(REINHARD, DRAGO, MANTIUK) 톤맵핑 기술을 이용해서 hdr을 직접 코드로 구현
- 각각의 톤맵핑 기술을 통해 얻은 hdr 영상의 히스토그램을 분석 
- 직접 찍은 영상의 hdr영상을 얻을 수 있는 코드

### corner.cpp파일
- blob-detector를 사용하여 영상의 동전을 찾아내고 개수를 찾아내는 코드
- 그림판으로 그린 도형들에서 corner-detector를 찾아내어 도형이 어떤 다각형인지 알아내는 코드
- SIFT특징 검출기를 이용해서 warping이나 밝기변화를 해도 영상의 SIFT특징점이 같게 검출이 되는지 확인해보는 코드

### Flip.cpp파일
- 영상을 여러가지 회전 변환 matrix를 통해 변환시켜보는 코드
- 카드 영상의 꼭짓접을 blob-detector를 통해 검출하여 기울어진 카드를 직사각형으로 perspective-transform시키는 코드

### homography.cpp 파일
- 직접 찍은 영상들을 파노라마 시키는 코드
- 책 표지 영상을 얻은 후 여러가지 물체가 섞여 있는 영상에서 특정 책을 찾아내는 코드
- 특징점 매칭 SIFT특징점 검출 - bruteforce - ransac알고리즘 사용 
