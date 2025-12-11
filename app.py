import streamlit as st
import requests
from datetime import datetime
import re
from collections import Counter
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.graph_objects as go


# Configuración de la página
st.set_page_config(
    page_title="Asistente de Noticias IA",
    page_icon="🔍",
    layout="wide"
)

def search_news_api(query, num_results=15):
    """Busca noticias reales desde Google News"""
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=es&gl=CO&ceid=CO:es"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            from xml.etree import ElementTree as ET
            root = ET.fromstring(response.content)
            
            noticias = []
            for item in root.findall('.//item')[:num_results * 2]:
                try:
                    title = item.find('title').text if item.find('title') is not None else "Sin título"
                    link = item.find('link').text if item.find('link') is not None else "#"
                    description = item.find('description').text if item.find('description') is not None else "Sin descripción"
                    pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
                    
                    fecha_formateada = "Reciente"
                    if pub_date:
                        try:
                            fecha_obj = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %Z')
                            fecha_formateada = fecha_obj.strftime('%Y-%m-%d')
                        except:
                            try:
                                fecha_obj = datetime.strptime(pub_date[:25], '%a, %d %b %Y %H:%M:%S')
                                fecha_formateada = fecha_obj.strftime('%Y-%m-%d')
                            except:
                                fecha_formateada = datetime.now().strftime('%Y-%m-%d')
                    
                    source = "Google News"
                    if ' - ' in title:
                        parts = title.split(' - ')
                        if len(parts) > 1:
                            source = parts[-1]
                            title = ' - '.join(parts[:-1])
                    
                    import html
                    clean_desc = re.sub('<[^<]+?>', '', description)
                    clean_desc = html.unescape(clean_desc)[:300]
                    
                    noticias.append({
                        "title": title,
                        "url": link,
                        "description": clean_desc,
                        "date": fecha_formateada,
                        "source": source
                    })
                    
                    if len(noticias) >= num_results:
                        break
                        
                except Exception as e:
                    continue
            
            return noticias if noticias else get_example_news(query)
        
        return get_example_news(query)
        
    except Exception as e:
        return get_example_news(query)

def get_example_news(query):
    """Datos de ejemplo si la API falla"""
    return [
        {
            "title": f"Últimos avances en {query}",
            "url": "https://ejemplo.com/noticia1",
            "description": f"Investigadores presentan nuevos desarrollos en {query} que podrían revolucionar la industria.",
            "date": "2025-12-08",
            "source": "Tech News"
        },
        {
            "title": f"El impacto de {query} en la sociedad",
            "url": "https://ejemplo.com/noticia2",
            "description": f"Expertos analizan cómo {query} está transformando diversos sectores económicos y sociales.",
            "date": "2025-12-07",
            "source": "Science Daily"
        }
    ]

def extract_keywords(texts, top_n=5):
    """Extrae palabras clave de los textos"""
    full_text = " ".join(texts).lower()
    
    stop_words = {'el', 'la', 'de', 'en', 'y', 'a', 'los', 'las', 'un', 'una', 'por', 'para', 
                  'con', 'del', 'al', 'es', 'que', 'se', 'su', 'como', 'más', 'sobre', 'este',
                  'esta', 'han', 'sido', 'son', 'está', 'desde', 'hace', 'muy', 'también'}
    
    words = re.findall(r'\b[a-záéíóúñ]{4,}\b', full_text)
    words = [w for w in words if w not in stop_words]
    
    word_counts = Counter(words)
    return [word for word, count in word_counts.most_common(top_n)]

def analyze_sentiment_simple(texts):
    """Análisis de sentimiento simple basado en palabras clave"""
    positive_words = ['éxito', 'avance', 'mejora', 'innovación', 'crecimiento', 'positivo', 
                      'beneficio', 'progreso', 'logro', 'desarrollo', 'excelente', 'bueno',
                      'aumenta', 'gana', 'victoria', 'revoluciona', 'mejorando']
    
    negative_words = ['crisis', 'problema', 'falla', 'fracaso', 'declive', 'negativo',
                      'perdida', 'riesgo', 'amenaza', 'crítica', 'deterioro', 'malo',
                      'disminuye', 'pierde', 'derrota', 'colapso', 'empeorando']
    
    full_text = " ".join(texts).lower()
    
    pos_count = sum(full_text.count(word) for word in positive_words)
    neg_count = sum(full_text.count(word) for word in negative_words)
    
    total = pos_count + neg_count + 1
    pos_percentage = int((pos_count / total) * 100)
    neg_percentage = int((neg_count / total) * 100)
    
    if pos_count > neg_count * 1.5:
        return "Positivo", 70 + pos_percentage
    elif neg_count > pos_count * 1.5:
        return "Negativo", 30 - neg_percentage
    else:
        return "Neutral", 50

def generate_summary_simple(texts, query):
    """Genera un resumen mejorado extrayendo información relevante"""
    if not texts:
        return f"No se encontraron noticias para generar un resumen sobre {query}."
    
    # Unir textos - usar más contenido
    full_text = " ".join(texts[:15])
    
    # Eliminar nombres de fuentes de forma segura
    fuentes = ['ITSitio', 'Infobae', 'BBC', 'CNN', 'Reuters', 'EFE', 'ABC', 
               'teleSUR', 'The Conversation', 'Semana', 'Portafolio', 'El Tiempo']
    
    for fuente in fuentes:
        # Solo eliminar cuando esté solo o seguido de punto
        full_text = re.sub(rf'\s+{fuente}\s*\.', '.', full_text)
        full_text = re.sub(rf'\s+{fuente}\s+', ' ', full_text)
    
    # Limpiar formato básico
    full_text = re.sub(r'\.\.+', '.', full_text)
    full_text = re.sub(r'\s+', ' ', full_text)
    
    # Dividir en oraciones por puntos
    sentences = []
    for sent in full_text.split('.'):
        sent = sent.strip()
        
        # Filtros de calidad
        if len(sent) < 50 or len(sent) > 300:
            continue
        if sent.count(' ') < 6:
            continue
        
        # Validar que sea una oración completa (tiene verbos probables)
        palabras_comunes = sent.lower().split()
        if not any(p in palabras_comunes for p in ['es', 'son', 'está', 'están', 'presenta', 'anuncia', 'indica', 'según', 'para', 'con', 'como']):
            continue
        
        # Debe mencionar el tema
        query_lower = query.lower()
        if not any(word in sent.lower() for word in query_lower.split()):
            continue
        
        sentences.append(sent.strip())
    
    if not sentences:
        return f"Se encontraron {len(texts)} noticias sobre {query}. Revisa las fuentes para más información."
    
    # Eliminar duplicados exactos y muy similares
    unique_sentences = []
    
    for sent in sentences:
        # Normalizar para comparar
        norm = sent.lower()
        
        # Verificar si ya existe algo muy similar
        is_dup = False
        for existing in unique_sentences:
            existing_norm = existing.lower()
            
            # Si comparten más del 70% de las palabras, es duplicado
            sent_words = set(norm.split())
            exist_words = set(existing_norm.split())
            
            if len(sent_words) > 0:
                overlap = len(sent_words & exist_words) / len(sent_words)
                if overlap > 0.7:
                    is_dup = True
                    break
        
        if not is_dup:
            unique_sentences.append(sent)
    
    if not unique_sentences:
        return f"Se analizaron {len(texts)} artículos sobre {query}."
    
    # Scoring por relevancia
    query_words = set(query.lower().split())
    scored = []
    
    for sent in unique_sentences[:15]:
        sent_words = set(sent.lower().split())
        score = len(query_words & sent_words)
        
        # Bonus por longitud apropiada
        if 80 < len(sent) < 200:
            score += 1
        
        scored.append((sent, score))
    
    # Ordenar por score
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Tomar las 4-5 mejores para un resumen más completo
    num_sentences = min(5, len(scored))  # Hasta 5 oraciones
    best = [s[0] for s in scored[:num_sentences]]
    
    # Construir resumen
    summary = ". ".join(best)
    
    if not summary.endswith('.'):
        summary += "."
    
    return summary

def get_recommendations(noticias, current_index, top_n=3):
    """Genera recomendaciones basadas en similitud semántica"""
    if len(noticias) <= 1:
        return []
    
    try:
        # Crear textos para análisis
        texts = [f"{n['title']} {n['description']}" for n in noticias]
        
        # Calcular TF-IDF
        vectorizer = TfidfVectorizer(max_features=100, stop_words=None)
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Calcular similitud coseno
        similarities = cosine_similarity(tfidf_matrix[current_index:current_index+1], tfidf_matrix)[0]
        
        # Obtener índices de las más similares (excluyendo la actual)
        similar_indices = similarities.argsort()[::-1][1:top_n+1]
        
        # Retornar noticias recomendadas
        recommendations = [noticias[i] for i in similar_indices if i < len(noticias)]
        
        return recommendations
    except:
        return []

def extract_related_topics(keywords, query):
    """Genera temas relacionados basados en palabras clave"""
    related = []
    query_lower = query.lower()
    
    # Palabras que sugieren temas relacionados
    tech_words = ['tecnología', 'digital', 'innovación', 'desarrollo']
    impact_words = ['impacto', 'efecto', 'consecuencia', 'resultado']
    future_words = ['futuro', 'predicción', 'tendencia', 'próximo']
    
    for keyword in keywords:
        if keyword not in query_lower:
            # Crear variaciones
            related.append(f"{query} {keyword}")
    
    # Sugerencias contextuales
    if any(word in query_lower for word in ['inteligencia', 'ia', 'artificial']):
        related.extend([f"{query} aplicaciones", f"{query} ética", f"{query} futuro"])
    
    if any(word in query_lower for word in ['cambio', 'clima', 'ambiental']):
        related.extend([f"{query} soluciones", f"{query} impacto", f"{query} políticas"])
    
    # Limitar a 5 sugerencias únicas
    return list(set(related))[:5]

# Título y descripción

# Header mejorado con estilo
st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='color: white; margin: 0;'>🔍 Asistente Inteligente de Noticias</h1>
        <p style='color: #f0f0f0; font-size: 18px; margin: 10px 0 0 0;'>Mantente actualizado con tecnología de Inteligencia Artificial</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Sección de búsqueda
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    query = st.text_input(
        "Ingresa tu tema de interés:",
        placeholder="Ej: inteligencia artificial, biotecnología, cambio climático...",
        key="search_query"
    )

with col2:
    num_results = st.number_input(
        "Número de noticias:",
        min_value=1,
        max_value=100,
        value=15,
        step=1,
        help="Ingresa cuántas noticias deseas analizar (máximo 100)"
    )
with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    search_button = st.button("🔍 Buscar", type="primary", use_container_width=True)

st.markdown("**Ejemplos:** `inteligencia artificial` · `tecnología cuántica` · `energías renovables` · `neurociencia`")

st.markdown("---")

# Procesar búsqueda
if search_button and query:
    num_to_fetch = num_results  # Usar el valor directamente del input
    
    with st.spinner("🔄 Buscando noticias en la web..."):
        noticias = search_news_api(query, num_results=num_to_fetch)
        
        if noticias:
            texts = [f"{n['title']}. {n['description']}" for n in noticias]
            
            with st.spinner("🤖 Analizando contenido..."):
                sentiment_label, sentiment_score = analyze_sentiment_simple(texts)
                keywords = extract_keywords(texts)
                resumen = generate_summary_simple(texts, query)
            
            st.session_state['noticias'] = noticias
            st.session_state['query'] = query
            st.session_state['sentiment'] = (sentiment_label, sentiment_score)
            st.session_state['keywords'] = keywords
            st.session_state['resumen'] = resumen
            
            st.success(f"✅ Se encontraron y analizaron {len(noticias)} noticias sobre '{query}'")

# Mostrar resultados
if 'noticias' in st.session_state:
    st.markdown("## 📊 Resultados del Análisis")
    
    col1, col2, col3 = st.columns(3)
    
    sentiment_label, sentiment_score = st.session_state.get('sentiment', ('Neutral', 50))
    keywords = st.session_state.get('keywords', [])
    
    with col1:
        st.metric(
            label="📰 Noticias Analizadas",
            value=len(st.session_state['noticias'])
        )
    
    with col2:
        st.metric(
            label="📈 Sentimiento",
            value=sentiment_label,
            delta=f"{sentiment_score}%"
        )
    
    with col3:
        st.metric(
            label="🔑 Palabras Clave",
            value=len(keywords),
            delta="extraídas"
        )
    
    if keywords:
        st.markdown("**🔑 Palabras clave identificadas:**")
        # Crear badges coloridos para las palabras clave
        keyword_html = ""
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        for i, keyword in enumerate(keywords):
            color = colors[i % len(colors)]
            keyword_html += f'<span style="background-color: {color}; color: white; padding: 5px 15px; border-radius: 20px; margin: 5px; display: inline-block; font-weight: bold;">{keyword}</span>'
        st.markdown(keyword_html, unsafe_allow_html=True)
    
    st.markdown("---")

    st.markdown("---")
    
    # Visualización de sentimiento
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📊 Distribución de Sentimiento")
        
        # Crear gráfico de dona
        sentiment_label, sentiment_score = st.session_state.get('sentiment', ('Neutral', 50))
        
        colors_map = {
            'Positivo': '#00D26A',
            'Neutral': '#FFD93D',
            'Negativo': '#FF6B6B'
        }
        
        # Calcular valores para el gráfico
        if sentiment_label == 'Positivo':
            positive_val = abs(sentiment_score)
            neutral_val = 0
            negative_val = 100 - abs(sentiment_score)
            color = colors_map['Positivo']
        elif sentiment_label == 'Negativo':
            positive_val = 0
            neutral_val = 0
            negative_val = abs(sentiment_score)
            positive_val = 100 - abs(sentiment_score)
            color = colors_map['Negativo']
        else:  # Neutral
            positive_val = 0
            neutral_val = abs(sentiment_score)
            negative_val = 100 - abs(sentiment_score)
            color = colors_map['Neutral']
        
        fig = go.Figure(data=[go.Pie(
            labels=[sentiment_label, 'Otros'],
            values=[abs(sentiment_score), 100 - abs(sentiment_score)],
            hole=.4,
            marker_colors=[color, '#2d3748'],
            showlegend=False
        )])
        
        fig.update_layout(
            annotations=[dict(text=sentiment_label, x=0.5, y=0.5, font_size=20, showarrow=False)],
            height=250,
            margin=dict(t=0, b=0, l=0, r=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Análisis Temporal")
        st.caption("Distribución de noticias por fecha")
        
        # Contar noticias por fecha
        from collections import Counter
        fechas = [n.get('date', 'Sin fecha')[:10] for n in st.session_state['noticias']]
        fecha_counts = Counter(fechas)
        
        # Ordenar por fecha
        sorted_dates = sorted(fecha_counts.items())
        
        # Determinar cuántas fechas mostrar según cantidad de noticias
        num_noticias = len(st.session_state['noticias'])
        num_fechas_mostrar = min(15, len(sorted_dates))  # Máximo 15 fechas
        
        # Tomar las más recientes
        recent_dates = sorted_dates[-num_fechas_mostrar:]
        
        # Crear gráfico de barras
        fig2 = go.Figure(data=[
            go.Bar(
                x=[d[0] for d in recent_dates],
                y=[d[1] for d in recent_dates],
                marker_color='#4ECDC4',
                text=[d[1] for d in recent_dates],  # Mostrar números en las barras
                textposition='outside'
            )
        ])
        
        fig2.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Cantidad de noticias",
            height=300,  # Aumentado de 250
            margin=dict(t=20, b=60, l=40, r=20),
            showlegend=False,
            xaxis_tickangle=-45,  # Rotar etiquetas 45 grados
            yaxis=dict(range=[0, max([d[1] for d in recent_dates]) * 1.2])  # Espacio arriba
        )
        
        fig2.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Cantidad",
            height=250,
            margin=dict(t=20, b=40, l=40, r=20),
            showlegend=False
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
    # Resumen
    st.markdown("## 📝 Resumen Inteligente")
    with st.container():
        resumen = st.session_state.get('resumen', '')
        if resumen:
            st.info("🤖 **Resumen generado por IA**")
            st.markdown(resumen)
        else:
            st.warning("No se pudo generar el resumen.")
    
    st.markdown("---")
    
    # Fuentes
    st.markdown("## 🔗 Fuentes Consultadas")
    
    if st.session_state['noticias']:
        fechas = [n.get('date', '') for n in st.session_state['noticias'] if n.get('date')]
        if fechas:
            st.caption("📅 Noticias publicadas entre las fechas más recientes disponibles")
    
    for i, noticia in enumerate(st.session_state['noticias'], 1):
        fecha_str = noticia.get('date', 'Fecha no disponible')
        antiguedad = ""
        
        if fecha_str != 'Fecha no disponible' and fecha_str != 'Reciente':
            try:
                fecha_noticia = datetime.strptime(fecha_str, '%Y-%m-%d')
                hoy = datetime.now()
                diff = (hoy - fecha_noticia).days
                
                if diff == 0:
                    antiguedad = " 🔴 HOY"
                elif diff == 1:
                    antiguedad = " 🟡 AYER"
                elif diff <= 7:
                    antiguedad = f" 🟢 Hace {diff} días"
                elif diff <= 30:
                    antiguedad = f" 🔵 Hace {diff//7} semanas"
                else:
                    antiguedad = " ⚪ Hace más de un mes"
            except Exception as e:
                antiguedad = " 📅 Reciente"
        else:
            antiguedad = " 📅 Reciente"

        # Estilo de tarjeta
        card_style = """
            <style>
            .stExpander {
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                margin-bottom: 10px;
            }
            </style>
        """
        if i == 1:
            st.markdown(card_style, unsafe_allow_html=True)
        
        with st.expander(f"📄 {i}. {noticia['title']}{antiguedad}", expanded=(i==1)):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Descripción:** {noticia['description']}")
                st.markdown(f"📅 **Fecha de publicación:** {noticia['date']}")
                st.markdown(f"🌐 **Fuente:** {noticia['source']}")
            
            with col2:
                st.link_button("🔗 Leer artículo completo", noticia['url'], use_container_width=True)
    
    st.markdown("---")
    
# Recomendaciones
    st.markdown("## 💡 Recomendaciones")
    
    # Noticias relacionadas
    st.markdown("### 📰 Noticias Relacionadas")
    
    if len(st.session_state['noticias']) > 1:
        # Obtener recomendaciones basadas en la primera noticia
        recommendations = get_recommendations(st.session_state['noticias'], 0, top_n=3)
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**{i}. {rec['title']}**")
                        st.caption(f"📅 {rec['date']} | 🌐 {rec['source']}")
                    
                    with col2:
                        st.link_button("Ver →", rec['url'], use_container_width=True)                    
                    st.markdown("---")
        else:
            st.info("No se pudieron generar recomendaciones para este tema.")
    
    # Temas relacionados
    st.markdown("### 🔎 Temas Relacionados para Explorar")
    
    keywords = st.session_state.get('keywords', [])
    if keywords:
        related_topics = extract_related_topics(keywords, st.session_state['query'])
        
        if related_topics:
            cols = st.columns(min(3, len(related_topics)))
            for idx, topic in enumerate(related_topics[:3]):
                with cols[idx]:
                    if st.button(f"🔍 {topic}", use_container_width=True):
                        st.session_state['search_query'] = topic
                        st.rerun()
        else:
            st.info("Explora las palabras clave identificadas para refinar tu búsqueda.")
    else:
        st.info("Realiza una búsqueda para ver temas relacionados.")
        
else:
    st.info("👆 Ingresa un tema de interés y presiona 'Buscar' para comenzar")
    
    st.markdown("## ✨ Características")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔍 Búsqueda Inteligente
        - Búsqueda en tiempo real
        - Múltiples fuentes de noticias
        - Filtrado relevante
        
        ### 📝 Análisis con IA
        - Resumen automático
        - Análisis de sentimiento
        - Extracción de palabras clave
        """)
    
    with col2:
        st.markdown("""
        ### 💡 Recomendaciones
        - Noticias relacionadas
        - Temas sugeridos
        - Tendencias conectadas
        
        ### 🔗 Referencias Completas
        - Enlaces a fuentes originales
        - Información de publicación
        - Acceso directo a artículos
        """)

st.markdown("---")
st.markdown("Desarrollado usando Streamlit y técnicas de IA")
