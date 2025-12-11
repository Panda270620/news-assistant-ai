# 🔍 Asistente Inteligente de Noticias IA

Aplicación web que utiliza Inteligencia Artificial para buscar, analizar y resumir noticias sobre cualquier tema de interés.

## 🎯 Características

- **🔍 Búsqueda inteligente**: Busca noticias en tiempo real sobre cualquier tema
- **📝 Resumen con IA**: Genera resúmenes automáticos usando modelos de lenguaje
- **📊 Análisis de sentimiento**: Analiza el tono general de las noticias
- **💡 Recomendaciones**: Sistema de recomendaciones basado en similitud semántica
- **🔗 Referencias completas**: Enlaces directos a todas las fuentes consultadas

## 🚀 Instalación

### Paso 1: Clonar el repositorio (o crear los archivos)

```bash
mkdir news-assistant-ai
cd news-assistant-ai
```

### Paso 2: Crear entorno virtual

```bash
python -m venv venv

# En Windows:
venv\Scripts\activate

# En Mac/Linux:
source venv/bin/activate
```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📦 Estructura del Proyecto

```
news-assistant-ai/
│
├── app.py                 # Aplicación principal de Streamlit
├── requirements.txt       # Dependencias del proyecto
├── README.md             # Este archivo
└── .gitignore            # Archivos a ignorar en git
```

## 🛠️ Tecnologías Utilizadas

- **Streamlit**: Framework para la interfaz web
- **Transformers (Hugging Face)**: Modelos de IA para NLP
- **Sentence Transformers**: Embeddings para recomendaciones
- **PyTorch**: Backend para modelos de deep learning

## 📅 Plan de Desarrollo

### ✅ Día 1 - Mañana (COMPLETADO)
- [x] Configurar entorno y dependencias
- [x] Implementar estructura básica de Streamlit
- [x] Crear interfaz de usuario

### 🔄 Día 1 - Tarde (EN PROGRESO)
- [ ] Integrar búsqueda web real
- [ ] Implementar modelo de resumen
- [ ] Implementar análisis de sentimiento

### 📝 Día 2 - Mañana
- [ ] Sistema de recomendaciones con embeddings
- [ ] Mejorar visualización de fuentes
- [ ] Optimizar rendimiento

### 🚀 Día 2 - Tarde
- [ ] Testing y ajustes finales
- [ ] Deploy a Streamlit Cloud
- [ ] Documentación final

## 🌐 Deploy en Streamlit Cloud

1. Sube tu proyecto a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio
4. ¡Listo! Tu app estará online

## 👨‍💻 Autor

Proyecto desarrollado para la materia de Inteligencia Artificial

## 📄 Licencia

MIT License - Libre uso educativo y personal
