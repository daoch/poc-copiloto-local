# POC Copiloto Empresarial Local

POC local con:

- Streamlit
- ChromaDB en contenedor
- Ollama en contenedor
- embeddings locales con `sentence-transformers`
- documentos locales de prueba

## Estructura

```text
poc-copiloto-local/
  docker-compose.yml
  Dockerfile
  requirements.txt
  app.py
  ingest.py
  rag.py
  logger.py
  documentos/
    rrhh/
    ti/
    ventas/
  data/
```

## Flujo esperado

1. Levantar servicios:

```powershell
docker compose up --build -d
```

2. Descargar el modelo local por defecto si aun no esta en el volumen:

```powershell
docker exec -it poc_ollama ollama pull phi3
```

3. Indexar documentos si es la primera corrida, si cambiaste documentos o si reiniciaste la base vectorial:

```powershell
docker exec -it poc_copiloto_app python ingest.py
```

4. Abrir la app:

`http://localhost:8501`

## Orden de ejecucion despues de `docker compose down`

- Si solo hiciste `docker compose down`:
  `docker compose up -d` -> probar la app
- Si hiciste `docker compose down` y cambiaste documentos:
  `docker compose up -d` -> `docker exec -it poc_copiloto_app python ingest.py` -> probar la app
- Si hiciste `docker compose down` y ademas borraste volumenes:
  `docker compose up -d` -> `docker exec -it poc_ollama ollama pull phi3` -> `docker exec -it poc_copiloto_app python ingest.py` -> probar la app

## Cambio opcional de modelo

Si quieres usar `llama3.1:8b`, cambia en `docker-compose.yml`:

```yaml
- OLLAMA_MODEL=llama3.1:8b
```

Luego vuelve a descargar el modelo y reinicia:

```powershell
docker exec -it poc_ollama ollama pull llama3.1:8b
docker compose down
docker compose up --build -d
```

## Limitacion actual

El rol del usuario solo se inyecta en el prompt. En esta version no existe filtrado real por permisos en ChromaDB, por lo que la recuperacion consulta toda la coleccion.

La siguiente iteracion deberia agregar metadata como `allowed_roles` y aplicar filtros en `collection.query(...)`.
