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
