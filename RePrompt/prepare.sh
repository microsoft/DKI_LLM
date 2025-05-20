cd ImageReward
pip install -e .
cd ..
cd Image-Generation-Cot/geneval
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -v -e .
cd /scratch/mingrui
pip uninstall opencv-python
pip uninstall opencv-python-headless
pip install -r requirements.txt
pip install openai azure-identity-broker --upgrade
pip install opencv-python-headless
python -m spacy download en_core_web_sm
cd trl
pip install -e .