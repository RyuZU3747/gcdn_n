# Philadelphia
<p align ='center'>
    <img src='./media/banner.webp' width='50%'>
</p>
프로젝트명 : 필라델피아
해당 코드는 필라델피아를 위해서 작성되었습니다.

코드의 사용법을 최대한 자세히 설명하겠지만, 잘 안되는 부분이 있다면 편한 방법으로 연락바랍니다.(ex. 깃허브 이슈, 이메일, 기타 등등)

special thanks : Jingeun Lee
## 구성목록
### Code 부분

아래는 코드의 대략적인 동작입니다. 자세한 내용은 doc에 기술되어 있습니다.<br>

- convertJson2excel.py
- getDataFromOpenPose.py
- plotPosOnVideo.py

#### convertJson2excel.py
OpenPose를 통해 얻은 프레임별 Json 파일에서 2d joint pos만을 추출해서 엑셀의 형태로 저장하는 코드입니다.

#### getDataFromOpenPose.py
파이썬 기본 모듈 별로 커널 호출 시, 동작 방식의 차이 때문에 가장 고생했던 코드입니다.<br>
해당 코드를 실행할 때, OpenPoseDemo.exe 파일 내부적으로 OpenPose 모델의 정의와 가중치에 대한 경로를 탐색하는데 visual studio 기준으로 터미널의 현재 위치가 중요합니다. 

#### plotPosOnVideo.py
excel file 형태로 저장된 joint position을 video에 plotting 하는 코드입니다.

----
일반적인 코드 실행의 흐름은 아래와 같습니다.
getDataFromOpenPose.py -> convertJson2excel.py -> plotPosOnVideo.py(선택사항)

### Dragon : Folder
해당 폴더는 OpenPose 및 Computer Vision 관련 작업을 할 때, 사용하면 아마 편리한 기능들이 정의되어 있습니다.
해당 프로젝트에서는 DragonV.py 모듈만 사용하시면 충분합니다.

해당 라이브러리의 의존성과 관련한 아래 리포지토리 링크 참고 바랍니다.<br> 
[Dragon_library](https://github.com/DaeeYong/dragon_library)<br>
(Star 눌러주면 좋겠다...)

#### 모듈 구성
- dcall.py
- dragonI.py
- dragonReadNet25.py
- DragonV.py


