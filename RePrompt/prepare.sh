cd ImageReward
pip install -e .
cd ..
pip uninstall opencv-python
pip uninstall opencv-python-headless
pip install -r requirements.txt
pip install opencv-python-headless
python -m spacy download en_core_web_sm
cd trl
pip install -e .
pip uninstall apex
cd ..