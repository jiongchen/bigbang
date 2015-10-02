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
