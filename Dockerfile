
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false PORT=8080
EXPOSE 8080
RUN mkdir -p /root/.streamlit &&     bash -lc 'cat > /root/.streamlit/config.toml <<EOF
[server]
headless = true
port = $PORT
address = "0.0.0.0"
enableCORS = true
enableXsrfProtection = true
EOF'
CMD ["bash","-lc","streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]
