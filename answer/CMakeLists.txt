file (GLOB_RECURSE answer_files "*.cxx")

find_package(benchmark REQUIRED)



foreach(src_file ${answer_files})
    get_filename_component(exe_name ${src_file} NAME_WE)
    add_executable(${exe_name} ${src_file})

    # target_compile_options(${exe_name} PRIVATE  -Werror -Wall -fsanitize=address)
    # target_link_options(${exe_name} PRIVATE -fsanitize=address)

    target_compile_options(${exe_name} PRIVATE  -Werror -Wall)

    target_link_libraries(${exe_name} benchmark::benchmark)

    target_include_directories(${exe_name} PRIVATE
		${CMAKE_SOURCE_DIR}/include
	)
endforeach()
