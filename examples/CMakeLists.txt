add_executable(test_unit test_unit.cc)
target_link_libraries(test_unit
    bigbang
    zjucad-ptree
    jtf-mesh
    ${Boost_LIBRARIES}
    lapack
)

add_executable(vol2tet vol2tet.cc)
target_link_libraries(vol2tet
    jtf-mesh
)

add_executable(test_pbd_cloth test_pbd_cloth.cc)
target_link_libraries(test_pbd_cloth
    bigbang
    ${Boost_LIBRARIES}
)

add_executable(test_simple simple_elastic_solver.cc)
target_link_libraries(test_simple
    bigbang
    ${Boost_LIBRARIES}
)

add_executable(binvox2hex binvox2hex.cc)

add_executable(lbfgs_demo lbfgs_demo.cc)
target_link_libraries(lbfgs_demo
    HLBFGS
)

add_executable(lbfgs_demo_other lbfgs_demo_other.cc)
target_link_libraries(lbfgs_demo_other
    ${LBFGS_LIBRARIES}
)

add_executable(test_mass_spring mass_spring.cc)
target_link_libraries(test_mass_spring
    bigbang
    ${Boost_LIBRARIES}
)

add_executable(test_embedded_fem embedded_fem.cc)
target_link_libraries(test_embedded_fem
    bigbang
    ${Boost_LIBRARIES}
)

add_executable(test_modal_basis test_modal_basis.cc)
target_link_libraries(test_modal_basis
    bigbang
    ${Boost_LIBRARIES}
)

add_executable(test_cloth test_cloth.cc)
target_link_libraries(test_cloth
    bigbang
    ${Boost_LIBRARIES}
)

add_executable(test_upsampling test_upsampling.cc)
target_link_libraries(test_upsampling
    bigbang
    ${Boost_LIBRARIES}
)

add_executable(test_proj_dyn test_proj_dyn.cc)
target_link_libraries(test_proj_dyn
    bigbang
    ${Boost_LIBRARIES}
)

add_executable(test_fast_proj test_fast_proj.cc)
target_link_libraries(test_fast_proj
    bigbang
    ${Boost_LIBRARIES}
)

add_executable(test_proj_dyn_tet test_proj_dyn_tet.cc)
target_link_libraries(test_proj_dyn_tet
    bigbang
    ${Boost_LIBRARIES}
)

add_executable(bin_reader bin_reader.cc)

add_executable(test_quasi_static test_quasi_static.cc)
target_link_libraries(test_quasi_static
    bigbang
    ${Boost_LIBRARIES}
)

add_executable(test_emb_thin_shell test_emb_thin_shell.cc)
target_link_libraries(test_emb_thin_shell
    bigbang
    ${Boost_LIBRARIES}
)

add_executable(read_obj read_obj.cc)
target_link_libraries(read_obj
    bigbang
    ${Boost_LIBRARIES}
)

add_executable(test_wrinkle_sim test_wrinkle_sim.cc)
target_link_libraries(test_wrinkle_sim
    bigbang
    ${Boost_LIBRARIES}
)

add_executable(test_alglib test_alglib.cc)
target_link_libraries(test_alglib
    ${ALGLIB_LIBRARIES}
)

add_executable(obj2vtk obj2vtk.cc)
target_link_libraries(obj2vtk
    jtf-mesh
)

add_executable(vtk2obj vtk2obj.cc)
target_link_libraries(vtk2obj
    jtf-mesh
)

add_executable(test_cosserat test_cosserat.cc)
target_link_libraries(test_cosserat
    bigbang
    ${Boost_LIBRARIES}
)

add_executable(test_frm_strain_rate test_frm_strain_rate.cc)
