using PyCall
@pyimport pip
pip.main(["install", "-r", "py-mpc/requirements.txt"])
if is_linux()
    cd("$(ENV["HOME"])/apps/gurobi701/linux64") do
        run(`$(PyCall.python) setup.py install`)
    end
else
    cd("/Library/gurobi702/mac64") do
        run(`$(PyCall.python) setup.py install`)
    end
end
