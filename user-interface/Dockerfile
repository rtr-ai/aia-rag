FROM node:22 as build

ARG LLM_ENDPOINT

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

RUN npm install --save-dev @types/node


#ENV LLM_ENDPOINT=${LLM_ENDPOINT}


RUN npm run build -- --output-path=dist --configuration=production

FROM nginx:alpine



ENV HOST=127.0.0.1

COPY nginx.conf /etc/nginx/nginx.conf
COPY --from=build /app/dist/* /usr/share/nginx/html

COPY docker-entrypoint.sh /

EXPOSE 8080
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]