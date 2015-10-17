enable_language(Fortran)

file(GLOB header *.h)
file(GLOB source *.cc)

set(SRC_DIR "{${CMAKE_CURRENT_SOURCE_DIR},$ENV{HOME}/.maxima}")

add_custom_command(OUTPUT potential.f90
    COMMAND "maxima" ARGS --userdir=${SRC_DIR} -b potential.mac
    DEPENDS potential.mac
)

add_custom_command(OUTPUT pbd_cons.f90
    COMMAND "maxima" ARGS --userdir=${SRC_DIR} -b pbd_constraint.mac
    DEPENDS pbd_constraint.mac
)

add_custom_command(OUTPUT elasticity.f90
    COMMAND "maxima" ARGS --userdir=${SRC_DIR} -b elasticity.mac
    DEPENDS elasticity.mac
)

add_subdirectory(HLBFGS)

add_library(bigbang SHARED ${header} ${source} potential.f90 pbd_cons.f90 elasticity.f90)
target_link_libraries(bigbang
    jtf-mesh
    ${UMFPACK_LIBRARIES}
    HLBFGS
)
