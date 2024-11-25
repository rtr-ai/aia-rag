FROM node:22

WORKDIR /app

COPY package.json package-lock.json ./

RUN npm install

COPY . .

EXPOSE 3000

ENV HOST=127.0.0.1

CMD ["npm", "start"]
