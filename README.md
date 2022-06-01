# The SEKO Assignment: Efficient and Fair Assignment of Students to Multiple Seminars

The Python script in the folder 'sekoAssignmentTool' implements the SEKO assignment strategy which is a lottery procedure for the fair assignment of interested students to seminars. The seminars are often limited in terms of the maximum number of participants. Due to the place limitation, those seminars are often offered several times to serve the students' demands. Still, some seminars are more popular than others and it may not be possible to grant access to all interested students due to capacity limitations. The SEKO assignment strategy achieves the following goals: 
1. Efficiency by utilizing all available seminar places, 
2. Satisfying all students by trying to assign at least one seminar to each student, and 
3. Fairness by considering the number of assigned seminars per student. 

The assignment of the seminars with respect to the key objectives is formulated by means of integer linear programming (ILP). The scripts and documentation can be found in the folder 'MIP_implementation'.

## Folder 'sekoAssignmentTool'
An implementation of the SEKO tool is available as Python script and as executable for Windows OS. Example input and output files are provided. A detailed README is provided as markdown and HTML file to explain the usage of the script in a console.

Files:
* sekoAssignmentTool.py : Implementation of the SEKO assignment in Python
* sekoAssignmentTool.exe : Executable for the Windows OS 
* input.xlsx : sample input file of student requests to seminar
* output.xlsx : this output file is generated by the SEKO assignment tool
* README.html : Detailed explanation how to use the SEKO tool in practice (HTML)
* README.md : Detailed explanation how to use the SEKO tool in practice (markdown)


## Folder 'MIP_implementation'
The optimization problems are implemented in Python. An example visualization of the optimal assignments as a result of the mixed integer programming (MIP) approaches is available as Jupyter notebook. The output is also provided as HTML file. 

Files:
* mip_seko.ipynb : Jupyter notebook running the MIPs and visualize the results
* mip_seko.html : static HTML output of the Jupyter notebook
* input.xlsx : input file of student requests to seminar
* output.xlsx : output file of the SEKO assignment
* moduleSEKO.py : Python module implementing the MIPs and SEKO
