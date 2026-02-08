# rosa_raspberry_gpio
ROSPBERRY(4/5)にUBUNTU24.04をいれて、ROS2を導入、langchain + ROSAで自然言語で指示してロボットカーに指令する

## インストール
```
git clone https://github.com/takeofuture/rosa_raspberry_gpio.git
cd rosa_raspberry_gpio
pip install -r requirement_rosa.txt
```
/opt/ros/jazzy/setup.bash　がある前提で（ROS2のJAZZYがある前提で）
```
bash run.sh
```
もし別のバージョンをいれていたら以下のコマンドを実行してSTREAMLITで立ち上げてください
```
source {your_ros_path}/setup.bash
source {your_python_venv_path}/bin/activate
streamlit run app_rccam_rosa.py --server.address 0.0.0.0 --server.port 8501
```

