# RemixFlow — slim, easy-install image (DSP backend + full UI, no GPU needed).
# The React UI is bundled in the PyPI wheel, so this just installs and runs.
#
#   docker run --rm -p 8770:8770 fbobe3/remixflow
#   → open http://localhost:8770
#
# For the real ACE-Step generative backend (GPU), see Dockerfile.gpu.
FROM python:3.11-slim

LABEL org.opencontainers.image.title="RemixFlow" \
      org.opencontainers.image.description="AI music evolution platform — steer songs and turn them into endless Living Songs." \
      org.opencontainers.image.source="https://github.com/fbobe321/remixflow" \
      org.opencontainers.image.licenses="MIT"

# 'audio' extra adds librosa (tempo/key detection, time-stretch/pitch-shift).
# soundfile bundles libsndfile, so MP3/WAV/FLAC/OGG work with no system packages.
RUN pip install --no-cache-dir "remixflow[audio]==0.1.1"

ENV REMIXFLOW_DATA_DIR=/data
VOLUME ["/data"]
EXPOSE 8770

CMD ["remixflow", "serve", "--host", "0.0.0.0", "--port", "8770"]
