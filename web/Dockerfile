FROM node:20-alpine

ENV NODE_ENV=production \
    PORT=8080

WORKDIR /app

COPY package*.json ./
RUN npm ci --omit=dev

COPY . .

EXPOSE 8080

CMD ["node", "server.js"]
