file(GLOB hlbfgs_header *.h)
file(GLOB hlbfgs_source *.cpp)

add_library(HLBFGS SHARED ${hlbfgs_source})
