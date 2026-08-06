# FoundationPose su Jetson Orin (JetPack 6.2.x / L4T R36.5.0)

Solo FoundationPose per ora: SAM3 e i VLM restano fuori da questo setup
(`Dockerfile.jetson` non li installa). Il `Dockerfile`/`compose.yml`
originali restano invariati per uso su macchina x86.

## Perche' non basta il Dockerfile originale

- `wenbowen123/foundationpose:latest` e' un'immagine x86_64 (non esiste
  per aarch64).
- Miniforge/torch vengono scaricati come wheel Linux-x86_64 / cu128,
  incompatibili con l'architettura ARM e la CUDA di Orin (sm_87).
- Le estensioni compilate da FoundationPose (mycpp, kaolin, nvdiffrast,
  pytorch3d) vanno ricompilate per sm_87.

## 1. Setup una tantum sull'Orin

### 1.1 Docker deve usare il runtime NVIDIA anche in fase di build
(serve la GPU per compilare le estensioni CUDA durante `docker build`,
non solo a runtime). Attualmente sul tuo Orin il default runtime e'
`runc`, va cambiato in `nvidia`:

```bash
sudo tee /etc/docker/daemon.json <<'EOF'
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
EOF
sudo systemctl restart docker
docker info | grep -i "Default Runtime"   # deve stampare: Default Runtime: nvidia
```

### 1.2 Installa jetson-containers

```bash
git clone https://github.com/dusty-nv/jetson-containers ~/jetson-containers
cd ~/jetson-containers
bash install.sh
```

### 1.3 Costruisci l'immagine base con gli stack pesanti gia' pronti per Orin

Questo comando concatena le ricette ufficiali (gia' testate per
JetPack 6.x / sm_87) di pytorch, torchvision, pytorch3d, kaolin e
nvdiffrast in una sola immagine:

```bash
jetson-containers build --name=foundationpose-base pytorch torchvision pytorch3d kaolin nvdiffrast
```

Puo' richiedere parecchio tempo alla prima esecuzione (compila diverse
estensioni CUDA). Al termine verifica il tag prodotto:

```bash
docker images | grep foundationpose-base
```

## 2. Build dell'immagine FoundationPose

Dalla cartella `perception_docker`, passando il tag trovato sopra:

```bash
cd perception_docker
BASE_IMAGE=foundationpose-base:<tag-trovato-sopra> \
  docker compose -f compose.jetson.yml build
```

oppure con `docker build` diretto:

```bash
docker build -f Dockerfile.jetson \
  --build-arg BASE_IMAGE=foundationpose-base:<tag> \
  -t perception-jetson .
```

## 3. Run

```bash
BASE_IMAGE=foundationpose-base:<tag> docker compose -f compose.jetson.yml up -d
docker compose -f compose.jetson.yml exec perception bash
# dentro al container:
run_foundationpose bash
```

## Note / cose da controllare al primo build

- `Dockerfile.jetson` NON esegue `build_all.sh` di FoundationPose
  cosi' com'e': quello script rifà da zero kaolin (`cd /kaolin && pip
  install -e .`), che qui arriva gia' pronto dall'immagine base. Viene
  eseguito solo lo step di build di `mycpp`.
- La lista finale di `pip install` (scipy, open3d, pyrender, ecc.) e'
  presa dal Dockerfile x86 originale di FoundationPose. Alcuni di questi
  pacchetti potrebbero non avere ancora una wheel precompilata per
  aarch64: se il build si ferma su uno di questi, commentalo/isolalo,
  builda il resto, e valuta se serve davvero per la pipeline che usi
  (es. `pyrender`/`meshcat` servono solo per visualizzazione debug).
- I pesi dei modelli (`weights/`) e i dati demo non sono nell'immagine:
  vanno montati o scaricati a parte, come nel setup x86.
