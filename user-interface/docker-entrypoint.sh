#!/bin/sh
echo "Replacing environment variables"

for file in /usr/share/nginx/html/*/assets/env.js /usr/share/nginx/html/assets/env.js;
do
    if [ -f "$file" ]; then
        echo "Processing $file..."
        sed -i "s|%FRIENDLY_CAPTCHA_SITEKEY%|$FRIENDLY_CAPTCHA_SITEKEY|g" $file
    fi
done

# Start Nginx
echo "Starting nginx"
exec nginx -g 'daemon off;'