#pragma once

namespace domainUtils {
    inline int index(int i, int j, int nvars) {
        return i * nvars + j;
}