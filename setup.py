from setuptools import setup,find_packages

HYPEN_E_DOT = "-e ."

requirements = []
def get_requirements(path):
    with open(path) as file:
        requirements = file.readlines()

        requirements = [requirement.replace('\n',"") for requirement in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements




setup(
    name= 'Name_Entity_Recognition',
    version='0.0.1',
    author='GnanaChaithanya',
    author_email='m.gnanachaithanya12@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements('Requirements.txt'))