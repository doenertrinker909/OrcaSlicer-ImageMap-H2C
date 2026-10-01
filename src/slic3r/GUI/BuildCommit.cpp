#include "BuildCommit.hpp"
#ifdef GIT_COMMIT_HASH
#undef GIT_COMMIT_HASH
#endif
#ifdef GIT_COMMIT_SUFFIX
#undef GIT_COMMIT_SUFFIX
#endif
#include "git_commit_hash.h"

namespace Slic3r { namespace GUI {

const char *const build_commit_hash  = GIT_COMMIT_HASH;
const char *const build_commit_label = GIT_COMMIT_HASH GIT_COMMIT_SUFFIX;

}} // namespace Slic3r::GUI
