file (GLOB_RECURSE answer_files "*.cxx")

foreach(src_file ${answer_files})
    get_filename_component(exe_name ${src_file} NAME_WE)
    add_executable(${exe_name} ${src_file})

    target_compile_options(${exe_name} PRIVATE  -Werror -Wall)

    target_include_directories(${exe_name} PRIVATE
		${CMAKE_SOURCE_DIR}/include
	)
endforeach()
