# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr whats_for_dinner-*.dist-info
	@rm -fr whats_for_dinner.egg-info

install:
	@pip install . -U

run_streamlit:
	streamlit run app.py

all: clean install
