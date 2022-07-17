# ViTPose Docker

## Build

```bash
sudo pacman -Sy docker
sudo groupadd docker
sudo usermod -aG docker $USER
paru -Sy nvidia-container-toolkit
sudo systemctl enable --now docker
docker build -t vitpose .
```
