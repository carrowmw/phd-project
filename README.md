The app.yaml file serves as a crucial configuration component for applications deployed on platforms like Google App Engine. It defines the runtime environment, resource allocation, scaling options, and other deployment settings specific to the application. This file allows developers to specify environment variables, handler scripts, security settings, and more, ensuring the application runs smoothly in the deployed environment. By customizing app.yaml, you can tailor the deployment to meet the application's requirements, such as managing traffic, integrating services, and optimizing performance.

When setting up a new virtual environment add export PYTHONPATH=$PYTHONPATH:/Users/administrator/Code/python/phd/configs to the activate script in the bin directory in the virtualenv so that PYTHONPATH will includes the project root directory, allowing Python to locate the relevant module in the project structure.

# Git

Main Branch: Initially, your repository will have a default branch (typically main). This branch should contain stable and production-ready code.

Development Branch: Create a develop branch from main for development work and feature integration. Use git checkout -b develop to create and switch to it.

Feature Branches: For new features, create branches off develop. Name them meaningfully, e.g., feature/new-data-processing.

Merge Requests: Merge feature branches back into develop after completion and review. Once develop is stable, merge it into main.