ARG VIDIOAI_VERSION=dev

FROM node:22-alpine AS dependencies
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:22-alpine AS builder
ARG VIDIOAI_VERSION
WORKDIR /app
COPY --from=dependencies /app/node_modules ./node_modules
COPY . .
# Une valeur vide signifie « même origine » et reste portable entre clouds.
ARG NEXT_PUBLIC_API_URL=""
ARG NEXT_PUBLIC_VIDIOAI_VERSION=${VIDIOAI_VERSION}
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL \
    NEXT_PUBLIC_VIDIOAI_VERSION=$NEXT_PUBLIC_VIDIOAI_VERSION \
    VIDIOAI_VERSION=$VIDIOAI_VERSION
RUN npm run build

FROM node:22-alpine AS runtime
ARG VIDIOAI_VERSION
ARG NEXT_PUBLIC_VIDIOAI_VERSION=${VIDIOAI_VERSION}
LABEL org.opencontainers.image.version="${VIDIOAI_VERSION}"
WORKDIR /app
ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0 \
    VIDIOAI_VERSION=${VIDIOAI_VERSION} \
    NEXT_PUBLIC_VIDIOAI_VERSION=${NEXT_PUBLIC_VIDIOAI_VERSION}
RUN addgroup --system --gid 10001 nodejs && adduser --system --uid 10001 nextjs
COPY --from=builder --chown=nextjs:nodejs /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
USER nextjs
EXPOSE 3000
CMD ["node", "server.js"]
