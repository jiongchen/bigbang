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

add_custom_command(OUTPUT stvk_hess_stiff_tensor.f90
    COMMAND "maxima" ARGS --userdir=${SRC_DIR} -b stvk_hess_stiff_tensor.mac
    DEPENDS stvk_hess_stiff_tensor.mac
)

add_subdirectory(HLBFGS)

if(CUDA_FOUND)
    cuda_add_library(cu-jacobi SHARED jacobi.cu)
    set(CU_JACOBI_LIBRARY cu-jacobi)
else(CUDA_FOUND)
    set(CU_JACOBI_LIBRARY "")
endif(CUDA_FOUND)

add_library(bigbang SHARED ${source} potential.f90 pbd_cons.f90 elasticity.f90 nnls.c)
target_link_libraries(bigbang
    jtf-mesh
    ${UMFPACK_LIBRARIES}
    HLBFGS
    ${LBFGS_LIBRARIES}
    ${ARPACK_LIBRARIES}
    ${CU_JACOBI_LIBRARY}
)
