## train and export

```bash
pip install -r requirements.txt
python lr_train_and_export.py models/lr
python lr_train_and_export.py models/lr2 # test to serve multiple models
```

## run on docker

```bash
docker-compose build
docker-compose up
```

## run on k8s

Firstly, build your docker image, push it to container registry, and edit image name in deploy.yml, then

```bash
kubectl apply -f deploy.yml
```