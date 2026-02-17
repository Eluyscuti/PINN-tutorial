# PINN-tutorial
This is a tutorial on how to train a forward PINN, and inverse PINN for the massdamper  (underdamped scenario); it also provides insight into the role of learning rates, and weights, and different activation functions and weights, and shows how sensitive the model is to such decisions. 
# Steps to run the tutorial
1. Set up an environment (conda, venv, or other virtual environment).
2. (Optional) Activate your environment. Example for the included `myenv` virtualenv:

	```bash
	source myenv/bin/activate
	```

3. Install the requirements:

	```bash
	python -m pip install --upgrade pip wheel
	python -m pip install -r requirements.txt
	```

4. Run the tutorial notebook `mass-damper-PINN-tutorial-beta.ipynb`.
