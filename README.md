# Description

This project, phd-project, is (will be) a comprehensive programme designed for building, testing, and deploying machine learning models fed on urban observatory data. Follow the instructions below to set up and run the project.

# Installation

## Prerequisites
Before you begin, ensure you have the following installed on your system:

Python (version 3.10 or higher recommended)
Poetry for dependency management and packaging
(Optional) pyenv for managing multiple Python versions

Poetry for dependency management and packaging ![link](https://python-poetry.org/docs/).

## Step 1: Clone the Repository

First, clone the project repository to your local machine using Git. Open a terminal and run:

```
git clone https://github.com/carrowmw/phd-project
cd phd-project
```

## Step 2: (Optional) Setting Up Python with pyenv
If you prefer to manage multiple Python versions on your system, you can use pyenv to install and set a specific Python version for this project:

```
pyenv install 3.10.0  # Skip if already installed
pyenv local 3.10.0
```

## Step 3: Installing Dependencies with Poetry
With Poetry installed, set up the project's dependencies by running the following command in the project root directory

```
poetry install
```

## Step 4: Activating the Virtual Environment

After installing the dependencies, you can activate the project's virtual environment using Poetry:

```
poetry shell
```

## Step 5: (Optional) Installing Development Dependencies with Poetry

For developers and contributors looking to run tests or use development tools, ensure you've installed the project's development dependencies:

```
poetry install --with dev
```

This includes additional packages like pytest for testing and code2flow for generating flowcharts from your code.

To run tests, you can use the following command:

bash```
pytest
```

# Running the Project
After installing the dependencies and activating the virtual environment, you can now run the project as follows:


bash```
python src/pipeline.py
python apps/missing_data_dashboard.py
python main.py #implementation coming soon
```

Note: This guide assumes a working knowledge of Python and command-line interfaces. If you encounter any issues, refer to the official Poetry documentation for more detailed information.

# Contributing
Not open to contributors currently, but feel free to fork, updates to come soon.

# Apps

The app.yaml file serves as a crucial configuration component for applications deployed on platforms like Google App Engine. It defines the runtime environment, resource allocation, scaling options, and other deployment settings specific to the application. This file allows developers to specify environment variables, handler scripts, security settings, and more, ensuring the application runs smoothly in the deployed environment. By customising app.yaml, you can tailor the deployment to meet the application's requirements, such as managing traffic, integrating services, and optimising performance.


# Git

Main Branch: Contains stable and production-ready code.

Development Branch: For development work and feature integration. Use git checkout -b develop to create and switch to it.

Feature Branches: For new features, create branches off develop. Name them meaningfully, e.g., feature/new-data-processing.

Merge Requests: Merge feature branches back into develop after completion and review. Once develop is stable, merge it into main.