#!/bin/sh

# Replace environment variables in JavaScript files
echo "Replacing environment variables"
for file in /usr/share/nginx/html/assets/env.js;
do
  echo "Processing $file..."
  sed -i "s|%FRIENDLY_CAPTCHA_SITEKEY%|$FRIENDLY_CAPTCHA_SITEKEY|g" $file
done

# Start Nginx
echo "Starting nginx"
exec nginx -g 'daemon off;'