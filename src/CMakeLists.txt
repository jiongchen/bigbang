enable_language(Fortran)

file(GLOB header *.h)
file(GLOB source *.cc)

set(SRC_DIR "{${CMAKE_CURRENT_SOURCE_DIR},$ENV{HOME}/.maxima}")

add_custom_command(OUTPUT potential.f90
    COMMAND "maxima" ARGS --userdir=${SRC_DIR} -b potential.mac
    DEPENDS potential.mac
)

add_library(bigbang SHARED ${header} ${source} potential.f90)
