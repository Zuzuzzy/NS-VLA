<div align="center">

# NS-VLA: Hacia modelos neuro-simbolicos de visión, lenguaje y acción

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Página del proyecto](https://img.shields.io/badge/Project-Page-blue)](https://zuzuzzy.github.io/NS-VLA/)
[![Modelo](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/zuzuzzy/NS-VLA)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/zuzuzzy/NS-VLA-Dataset)
[![Licencia](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

[**Página principal**](https://zuzuzzy.github.io/NS-VLA/) | [**Artículo**](https://arxiv.org/abs/XXXX.XXXXX) | [**Modelo**](https://huggingface.co/zuzuzzy/NS-VLA) | [**Dataset**](https://huggingface.co/datasets/zuzuzzy/NS-VLA-Dataset)
</div>

## Descripción general

NS-VLA es un nuevo marco de **Visión-Lenguaje-Acción Neuro-Simbólico** para la manipulación robótica. Introduce lo siguiente:

- 🧩 **Codificador neuro-simbólico**: Conecta el historial condicionado por instrucciones con un primitivo discreto, extrayendo un plan a nivel de episodio una sola vez y avanzando luego un puntero monótono a lo largo de él.
- ⚡ **Solucionador neuro-simbólico**: Conecta el primitivo activo a una política independiente del backbone mediante un puente condicionado por el primitivo, emitiendo acciones en bucle abierto en forma de fragmentos.
- 🔄 **Optimización conjunta jerárquica de políticas**: Actualiza ambos factores utilizando flujos de recompensa adaptados a su granularidad, combinando H-GRPO para la selección de primitivos con AWR para el control de fragmentos.

<p align="center">
  <img src="assets/pipeline.png" width="90%" alt="Marco NS-VLA"/>
</p>

## Rendimiento

Tasa de éxito (SR) en el entorno de un solo intento, promediada sobre las cuatro suites de LIBERO (en distribución) y sobre las suites perturbadas de LIBERO-Plus. La robustez es la relación entre la SR promedio de LIBERO-Plus y la SR promedio de LIBERO, expresada en porcentaje.

| Método | LIBERO (1-shot) | LIBERO-Plus (OOD) | Robustez |
|:---|:---:|:---:|:---:|
| Diffusion Policy | 17,2 | 6,0 | 34,9 |
| OpenVLA | 35,7 | 12,5 | 35,0 |
| π₀ | 37,4 | 12,5 | 33,4 |
| WorldVLA | 38,3 | 10,3 | 26,8 |
| NORA | 38,3 | 13,5 | 35,2 |
| OpenVLA-OFT | 48,9 | 18,5 | 37,8 |
| π₀-Fast | 50,0 | 20,5 | 41,0 |
| VLA-RL | 53,0 | 20,8 | 39,2 |
| UniVLA | 55,1 | 19,8 | 35,9 |
| SimpleVLA-RL | 55,3 | 24,8 | 44,8 |
| RIPT-VLA | 59,3 | 27,8 | 46,8 |
| VLA-Adapter | 65,3 | 28,0 | 42,9 |
| **NS-VLA (Nuestro método)** | **69,1** | **49,8** | **72,0** |

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/Zuzuzzy/NS-VLA.git
cd NS-VLA

# Crear un entorno Conda
conda create -n nsvla python=3.10 -y
conda activate nsvla

# Instalar dependencias
pip install -r requirements.txt
```

## Estructura del proyecto

```
NS-VLA/
├── configs/                  # Configuraciones de entrenamiento y evaluación
│   ├── train/
│   └── eval/
├── nsvla/                    # Framework principal de NS-VLA
│   ├── encoder/              # Codificador neuro-simbólico
│   ├── solver/               # Solucionador neuro-simbólico
│   ├── rl/                   # Optimización conjunta jerárquica de políticas
│   ├── primitives/           # Definiciones de primitivos y clasificador
│   ├── envs/                 # Envoltorios de entornos de referencia
│   ├── eval/                 # Esquema de rastreo y métricas
│   ├── train/                # Etapas de entrenamiento supervisado
│   └── utils/                # Funciones de utilidad
├── scripts/                  # Scripts de entrenamiento y evaluación
│   ├── train.sh
│   ├── eval.sh
│   └── annotate.sh
├── data/                     # Procesamiento de datos y primitivos
│   ├── prepare_1shot.py
│   ├── annotate_demos.py
│   └── extract_features.py
├── assets/                   # Figuras y medios
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## Inicio rápido

### Entrenamiento

```bash
# Etapa I: Preentrenamiento supervisado
bash scripts/train.sh --stage pretrain --config configs/train/pretrain.yaml

# Etapa II: Optimización conjunta jerárquica de políticas
bash scripts/train.sh --stage rl --config configs/train/rl_grpo.yaml
```

### Evaluación

```bash
# Evaluar en LIBERO
bash scripts/eval.sh --benchmark libero --checkpoint ruta/al/checkpoint

# Evaluar en LIBERO-Plus
bash scripts/eval.sh --benchmark libero_plus --checkpoint ruta/al/checkpoint
```

## Cita

Si considera que nuestro trabajo es útil, por favor cite lo siguiente:

```bibtex
@article{zhu2026nsvla,
  title={NS-VLA: Hacia modelos neuro-simbólicos de visión, lenguaje y acción},
  author={Zhu, Ziyue and Wu, Shangyang and Zhao, Shuai and Zhao, Zhiqiu and Li, Shengjie and Wang, Yi and Li, Fang and Luo, Haoran},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## Agradecimiento

Agradecemos a los desarrolladores de [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [CALVIN](https://github.com/mees/calvin), [OpenVLA](https://github.com/openvla/openvla), [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter) y [Qwen-VL](https://github.com/QwenLM/Qwen-VL) por sus contribuciones de código abierto.

## Licencia

Este proyecto está bajo licencia Apache 2.0. Consulte el archivo [LICENSE](LICENSE) para obtener más detalles.
