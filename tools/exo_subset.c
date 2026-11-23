/* Copy selected variables of an Exodus (netCDF) file, keeping only the time steps
 * with time_whole <= tmax. Needs only the netCDF C library (e.g. the netcdf-c module).
 *
 *   exo_subset in.e out.e tmax var [var ...]
 *
 * Example (T and the material/pore fraction for plot_temperature_area.py):
 *   exo_subset pore500hydro.e pore500hydro_TA.e 0.65 \
 *       name_nod_var coordx coordy time_whole vals_nod_var4 vals_nod_var12
 *
 * Build:
 *   gcc -O2 -o exo_subset tools/exo_subset.c $(nc-config --cflags) $(nc-config --libs)
 */
#include <netcdf.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call)                                                              \
    do {                                                                         \
        int status_ = (call);                                                    \
        if (status_ != NC_NOERR) {                                               \
            fprintf(stderr, "%s failed: %s\n", #call, nc_strerror(status_));     \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

static int create_mode(int format)
{
    switch (format) {
    case NC_FORMAT_64BIT_OFFSET: return NC_64BIT_OFFSET;
#ifdef NC_FORMAT_CDF5
    case NC_FORMAT_CDF5: return NC_64BIT_DATA;
#endif
    case NC_FORMAT_NETCDF4: return NC_NETCDF4;
    case NC_FORMAT_NETCDF4_CLASSIC: return NC_NETCDF4 | NC_CLASSIC_MODEL;
    default: return 0;
    }
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "Usage: %s in.e out.e tmax var [var ...]\n", argv[0]);
        return 1;
    }
    const char *inPath = argv[1], *outPath = argv[2];
    char *end;
    const double tmax = strtod(argv[3], &end);
    if (*end != '\0') {
        fprintf(stderr, "tmax must be a number, got '%s'\n", argv[3]);
        return 1;
    }

    int in, out, format, timeDim, timeVar;
    CHECK(nc_open(inPath, NC_NOWRITE, &in));
    CHECK(nc_inq_format(in, &format));
    CHECK(nc_inq_dimid(in, "time_step", &timeDim));
    CHECK(nc_inq_varid(in, "time_whole", &timeVar));
    size_t steps;
    CHECK(nc_inq_dimlen(in, timeDim, &steps));
    double *times = malloc((steps ? steps : 1) * sizeof(double));
    CHECK(nc_get_var_double(in, timeVar, times));
    size_t keep = 0;
    while (keep < steps && times[keep] <= tmax) ++keep;
    if (keep == 0) {
        fprintf(stderr, "no time step with time_whole <= %g\n", tmax);
        return 1;
    }

    const int nvars = argc - 4;
    int *inIds = malloc(nvars * sizeof(int)), *outIds = malloc(nvars * sizeof(int));
    CHECK(nc_create(outPath, NC_CLOBBER | create_mode(format), &out));
    for (int k = 0; k < nvars; ++k) {
        nc_type type;
        int ndims, dims[NC_MAX_VAR_DIMS], outDims[NC_MAX_VAR_DIMS];
        if (nc_inq_varid(in, argv[4 + k], &inIds[k]) != NC_NOERR) {
            fprintf(stderr, "variable '%s' not found in %s\n", argv[4 + k], inPath);
            return 1;
        }
        CHECK(nc_inq_var(in, inIds[k], NULL, &type, &ndims, dims, NULL));
        for (int d = 0; d < ndims; ++d) {
            char name[NC_MAX_NAME + 1];
            size_t length;
            CHECK(nc_inq_dim(in, dims[d], name, &length));
            if (nc_inq_dimid(out, name, &outDims[d]) != NC_NOERR)
                CHECK(nc_def_dim(out, name, dims[d] == timeDim ? NC_UNLIMITED : length, &outDims[d]));
        }
        CHECK(nc_def_var(out, argv[4 + k], type, ndims, outDims, &outIds[k]));
    }
    CHECK(nc_enddef(out));

    for (int k = 0; k < nvars; ++k) {
        nc_type type;
        int ndims, dims[NC_MAX_VAR_DIMS];
        size_t typeSize, start[NC_MAX_VAR_DIMS], count[NC_MAX_VAR_DIMS], values = 1;
        CHECK(nc_inq_var(in, inIds[k], NULL, &type, &ndims, dims, NULL));
        CHECK(nc_inq_type(in, type, NULL, &typeSize));
        int timeAxis = -1;
        for (int d = 0; d < ndims; ++d) {
            start[d] = 0;
            CHECK(nc_inq_dimlen(in, dims[d], &count[d]));
            if (dims[d] == timeDim) {
                timeAxis = d;
                count[d] = 1;  /* copied one step at a time */
            }
            values *= count[d];
        }
        void *buffer = malloc(values ? values * typeSize : 1);
        const size_t records = timeAxis < 0 ? 1 : keep;
        for (size_t r = 0; r < records; ++r) {
            if (timeAxis >= 0) start[timeAxis] = r;
            CHECK(nc_get_vara(in, inIds[k], start, count, buffer));
            CHECK(nc_put_vara(out, outIds[k], start, count, buffer));
        }
        free(buffer);
    }
    CHECK(nc_close(out));
    CHECK(nc_close(in));
    printf("%s: %zu of %zu steps (t = %g to %g), %d variables\n", outPath, keep, steps, times[0],
           times[keep - 1], nvars);
    free(times);
    free(inIds);
    free(outIds);
    return 0;
}
