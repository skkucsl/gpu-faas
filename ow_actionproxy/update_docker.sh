./gradlew :core:actionProxy:distDocker :sdk:docker:distDocker
docker tag whisk/dockerskeleton bskoon/$1
docker push bskoon/$1
