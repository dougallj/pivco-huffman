#!/bin/sh
# fetch_silesia_lits.sh — download the Silesia corpus and regenerate
# testdata/silesia-lits/*.lits (LZ4HC-9 literal streams).
#
# The committed .lits files are the reference; this script exists to
# reproduce them from the original corpus and verify the hashes.
# Requires: curl, unzip, cc, liblz4 (headers + lib; e.g. `brew install
# lz4` or `apt install liblz4-dev`).  Note the .lits were generated
# with liblz4 1.10.0 — LZ4HC's literal/match split may differ under
# other versions, in which case the committed files remain canonical.
#
# Usage: extras/fetch_silesia_lits.sh [workdir]   (default: ./silesia-tmp)
set -e

REPO=$(cd "$(dirname "$0")/.." && pwd)
WORK=${1:-./silesia-tmp}
OUT="$REPO/testdata/silesia-lits"
URL="https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip"
FILES="dickens mozilla mr nci ooffice osdb reymont samba sao webster x-ray xml"

mkdir -p "$WORK" "$OUT"
cd "$WORK"

missing=0
for f in $FILES; do [ -f "$f" ] || missing=1; done
if [ $missing = 1 ]; then
    if [ ! -f silesia.zip ]; then
        echo "fetching $URL"
        curl -fLO "$URL" || {
            echo "download failed — fetch silesia.zip manually into $WORK" >&2
            exit 1
        }
    fi
    unzip -oq silesia.zip
fi

# verify the corpus inputs if the hash file is present
if [ -f "$OUT/SHA256SUMS.inputs" ]; then
    if command -v shasum > /dev/null; then
        (cd "$WORK" && shasum -a 256 -c "$OUT/SHA256SUMS.inputs")
    else
        (cd "$WORK" && sha256sum -c "$OUT/SHA256SUMS.inputs")
    fi
fi

# build the extractor (pkg-config if available, else homebrew prefix, else system)
LZ4_CFLAGS=$(pkg-config --cflags liblz4 2>/dev/null || true)
LZ4_LIBS=$(pkg-config --libs liblz4 2>/dev/null || echo "-llz4")
if [ -z "$LZ4_CFLAGS" ] && command -v brew > /dev/null; then
    P=$(brew --prefix lz4 2>/dev/null || true)
    [ -n "$P" ] && LZ4_CFLAGS="-I$P/include" && LZ4_LIBS="-L$P/lib -llz4"
fi
cc -O2 $LZ4_CFLAGS "$REPO/extras/lz4_lits.c" $LZ4_LIBS -o lz4_lits

for f in $FILES; do
    ./lz4_lits "$f" "$OUT/$f.lits" > /dev/null
    echo "  $f.lits"
done

cd "$OUT"
if command -v shasum > /dev/null; then
    shasum -a 256 -c SHA256SUMS
else
    sha256sum -c SHA256SUMS
fi
echo "OK: 12 .lits regenerated and verified in $OUT"
