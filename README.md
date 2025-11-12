# PPE-YOLO — Detecção de EPIs em Tempo Real (Capacete + Óculos)

Este projeto implementa um sistema de **detecção de Equipamentos de Proteção Individual (EPIs)** usando modelos **YOLOv8/YOLO9**, com suporte a detecção em **tempo real via webcam**.

---

## 📦 Dataset Utilizado

O modelo utiliza o dataset **SH17 – PPE Detection Dataset**, disponível gratuitamente no Kaggle:

🔗 https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection

Este dataset contém mais de 15 mil instâncias anotadas, incluindo:
- Capacete (Helmet)
- Óculos de proteção (Glasses)
- Luvas (Gloves)
- Máscara (Mask)
- Colete de segurança (Safety Vest)
- Classes “NO-PPE” indicando ausência de EPI

O dataset foi criado e mantido por **Mughees Ahmad** e a equipe associada ao projeto.

---

## 📂 Estrutura do Projeto

```
project/
│── data/              # coloque aqui datasets (opcional)
│── models/
│   └── yolo9e.pt      # modelo pré-treinado
│── runs/              # saídas geradas automaticamente
│── detect_webcam.py   # script principal de detecção ao vivo
│── train.py           # opcional — treinar ou re-treinar modelos
│── requirements.txt   # dependências
└── README.md
```

---

## 🧠 Modelos Utilizados

### 🔹 YOLO9-E (recomendado)
- Melhor precisão (mAP50 ≈ 70.9%)
- Excelente para EPI (capacete, óculos, colete, etc.)
- Ideal para TCC e ambiente corporativo

Baixe o modelo pré-treinado na pasta **models/**:
```
models/yolo9e.pt
```

---

## ⚙️ Instalação

### 1. Crie o ambiente virtual
```
python -m venv .venv
```

### 2. Ative o ambiente
Windows:
```
.venv\Scripts\activate
```

### 3. Instale as dependências
```
pip install -r requirements.txt
```

---

## 🎥 Detecção em Tempo Real (Webcam)

Para rodar o detector ao vivo:

```
python detect_webcam.py --weights models/yolo9e.pt --device 0
```

Parâmetros úteis:

| Parâmetro | Exemplo | Descrição |
|----------|---------|-----------|
| `--device` | `0` | usa GPU |
| `--device` | `cpu` | usa CPU |
| `--conf` | `0.25` | confiança mínima |
| `--imgsz` | `960` | resolução melhor |

Exemplo completo:
```
python detect_webcam.py --weights models/yolo9e.pt --device 0 --conf 0.20 --imgsz 960
```

---

## 🏋️‍♂️ Treinando (opcional)

Caso deseje treinar seu próprio dataset:

### 1. Coloque o dataset dentro de `data/`:

Estrutura:
```
data/
 ├── train/
 ├── valid/
 └── data.yaml
```

### 2. Execute o treinamento
```
python train.py
```

Por padrão, o treinamento usa:
- YOLOv8-nano
- 50 épocas
- Resolução 640
- GPU (device 0)

---

## 📘 detect_webcam.py — Comportamento

O script:
- Lê a webcam
- Processa cada frame com YOLO
- Detecta **Helmet, Glasses e demais EPIs**
- Exibe caixa verde/vermelha
- Mostra status “EPI OK” ou “FALTANDO”

Ideal para monitoramento, TCC e demonstrações reais.

---

## 📝 Logs e Resultados

Tudo que o YOLO gerar (imagens, gráficos, métricas) será salvo automaticamente em:

```
runs/detect/
```

---

## 💡 Dicas de Performance

- Use `--imgsz 960` para melhor precisão.
- Use `YOLO9-E` para o melhor reconhecimento de óculos/capacetes.
- Caso o FPS caia, reduza para `--imgsz 640`.
- Em GPUs fracas, desabilite half precision:
  ```
  --half False
  ```

---

## 📄 Licença

Uso livre para fins educacionais, acadêmicos e corporativos internos.

---

## 👨‍💻 Autor

Projeto configurado com orientação assistida por IA.  
Integrado ao sistema de TCC do João — USCS.

