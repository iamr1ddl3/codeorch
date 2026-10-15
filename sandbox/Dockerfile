# CodeOrch build sandbox: a pinned Next.js 15 + TypeScript + Tailwind 4 app with
# dependencies pre-installed. Generated files are mounted read-only at /gen and
# overlaid at run time. Run with --network none (see runner.py) so generated
# code cannot reach the network or install packages.
FROM node:22-bookworm-slim

ENV NEXT_TELEMETRY_DISABLED=1
WORKDIR /app

COPY template/package.json template/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY template/ ./
COPY check.sh entrypoint.sh /usr/local/bin/
RUN chown -R node:node /app

USER node
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
