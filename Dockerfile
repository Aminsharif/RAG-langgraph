FROM langchain/langgraph-api:3.11







# -- Adding local package . --
ADD . /deps/RAG-chatbot
# -- End of local package . --



# -- Installing all local dependencies --

RUN for dep in /deps/*; do             echo "Installing $dep";             if [ -d "$dep" ]; then                 echo "Installing $dep";                 (cd "$dep" && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e .);             fi;         done

# -- End of local dependencies install --

ENV LANGGRAPH_STORE='{"type": "postgres", "connection": "postgresql://postgres:123456@langgraph-postgres:5432/langgraphrag?sslmode=disable"}'

ENV LANGGRAPH_AUTH='{"path": "/deps/RAG-chatbot/backend/security/auth.py:auth"}'

ENV LANGGRAPH_HTTP='{"app": "/deps/RAG-chatbot/backend/security/app/api/v1/endpoints/auth.py:app"}'

ENV LANGGRAPH_CHECKPOINTER='{"type": "postgres", "connection": "postgresql://postgres:123456@langgraph-postgres:5432/langgraphrag?sslmode=disable"}'

ENV LANGSERVE_GRAPHS='{"indexer": "/deps/RAG-chatbot/backend/agent/graph/graph_index.py:graph", "agent": "/deps/RAG-chatbot/backend/agent/graph/graph.py:graph"}'







# -- Ensure user deps didn't inadvertently overwrite langgraph-api
RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license && touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir --no-deps -e /api
# -- End of ensuring user deps didn't inadvertently overwrite langgraph-api --
# -- Removing build deps from the final image ~<:===~~~ --
RUN pip uninstall -y pip setuptools wheel
RUN rm -rf /usr/local/lib/python*/site-packages/pip* /usr/local/lib/python*/site-packages/setuptools* /usr/local/lib/python*/site-packages/wheel* && find /usr/local/bin -name "pip*" -delete || true
RUN rm -rf /usr/lib/python*/site-packages/pip* /usr/lib/python*/site-packages/setuptools* /usr/lib/python*/site-packages/wheel* && find /usr/bin -name "pip*" -delete || true
RUN uv pip uninstall --system pip setuptools wheel && rm /usr/bin/uv /usr/bin/uvx



WORKDIR /deps/RAG-chatbot