#!/usr/bin/env bash
# scripts/sync-consolidated.sh
# Verifies and synchronizes `consolidated` with its three upstream sources:
#   1. llama.cpp.ML-Eng (PrismML, branch: prism)
#   2. llama-cpp-turboquant (branch: feature/turboquant-kv-cache)
#   3. Mintplex-Prism-Orig (branch: prism)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSOLIDATED_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_DIR="$(cd "${CONSOLIDATED_DIR}/.." && pwd)"

ML_ENG_DIR="${BASE_DIR}/llama.cpp.ML-Eng"
ML_ENG_BRANCH="prism"

TURBO_DIR="${BASE_DIR}/llama-cpp-turboquant"
TURBO_BRANCH="feature/turboquant-kv-cache"

MINTPLEX_DIR="${BASE_DIR}/Mintplex-Prism-Orig"
MINTPLEX_BRANCH="prism"

ACTION="${1:---check}"

echo "=========================================================="
echo " Consolidated Repository Sync & Verification"
echo " Consolidated root: ${CONSOLIDATED_DIR}"
echo " Source base directory: ${BASE_DIR}"
echo "=========================================================="

cd "${CONSOLIDATED_DIR}"

# Ensure remotes exist
git remote add ml-eng "${ML_ENG_DIR}" 2>/dev/null || git remote set-url ml-eng "${ML_ENG_DIR}"
git remote add turboquant "${TURBO_DIR}" 2>/dev/null || git remote set-url turboquant "${TURBO_DIR}"
git remote add mintplex "${MINTPLEX_DIR}" 2>/dev/null || git remote set-url mintplex "${MINTPLEX_DIR}"

echo "[1/4] Fetching latest refs from local remotes..."
git fetch ml-eng "${ML_ENG_BRANCH}" --quiet
git fetch turboquant "${TURBO_BRANCH}" --quiet
git fetch mintplex "${MINTPLEX_BRANCH}" --quiet

ML_ENG_HEAD=$(git rev-parse ml-eng/"${ML_ENG_BRANCH}")
TURBO_HEAD=$(git rev-parse turboquant/"${TURBO_BRANCH}")
MINTPLEX_HEAD=$(git rev-parse mintplex/"${MINTPLEX_BRANCH}")
CONSOLIDATED_HEAD=$(git rev-parse HEAD)

echo
echo "--- Branch Status ---"
echo "consolidated HEAD: ${CONSOLIDATED_HEAD:0:10} ($(git log -1 --format='%s' HEAD))"
echo

check_ancestor() {
    local name="$1"
    local tip="$2"
    local branch="$3"
    local commit_msg
    commit_msg=$(git log -1 --format='%s' "${tip}")
    if git merge-base --is-ancestor "${tip}" HEAD; then
        echo -e "  [x] ${name} (${branch}): ${tip:0:10} - MERGED (up-to-date)\n      Commit: \"${commit_msg}\""
        return 0
    else
        local missing_count
        missing_count=$(git rev-list --count HEAD.."${tip}")
        echo -e "  [!] ${name} (${branch}): ${tip:0:10} - NOT MERGED (${missing_count} commits ahead)\n      Commit: \"${commit_msg}\""
        return 1
    fi
}

all_merged=true
check_ancestor "llama.cpp.ML-Eng" "${ML_ENG_HEAD}" "${ML_ENG_BRANCH}" || all_merged=false
echo
check_ancestor "llama-cpp-turboquant" "${TURBO_HEAD}" "${TURBO_BRANCH}" || all_merged=false
echo
check_ancestor "Mintplex-Prism-Orig" "${MINTPLEX_HEAD}" "${MINTPLEX_BRANCH}" || all_merged=false
echo

if [ "${all_merged}" = true ]; then
    echo "Result: Consolidated repository is fully synchronized with all three source repos."
    exit 0
fi

if [ "${ACTION}" != "--merge" ]; then
    echo "Notice: Unmerged commits found. Run with '--merge' to merge them into consolidated."
    exit 1
fi

echo "--- Merging pending branches into consolidated ---"
# Check working tree cleanliness
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: Working directory has uncommitted modifications. Please stash or commit them before running --merge."
    exit 1
fi

BACKUP_BRANCH="backup/consolidated-pre-sync-$(date +%Y%m%d-%H%M%S)"
git branch "${BACKUP_BRANCH}"
echo "Created backup branch: ${BACKUP_BRANCH}"

if ! git merge-base --is-ancestor "${ML_ENG_HEAD}" HEAD; then
    echo "Merging ml-eng/${ML_ENG_BRANCH}..."
    git merge ml-eng/"${ML_ENG_BRANCH}" -m "Merge branch 'ml-eng/${ML_ENG_BRANCH}' into consolidated"
fi

if ! git merge-base --is-ancestor "${TURBO_HEAD}" HEAD; then
    echo "Merging turboquant/${TURBO_BRANCH}..."
    git merge turboquant/"${TURBO_BRANCH}" -m "Merge branch 'turboquant/${TURBO_BRANCH}' into consolidated"
fi

if ! git merge-base --is-ancestor "${MINTPLEX_HEAD}" HEAD; then
    echo "Merging mintplex/${MINTPLEX_BRANCH}..."
    git merge mintplex/"${MINTPLEX_BRANCH}" -m "Merge branch 'mintplex/${MINTPLEX_BRANCH}' into consolidated"
fi

echo "Merge complete! Current HEAD: $(git rev-parse --short HEAD)"

