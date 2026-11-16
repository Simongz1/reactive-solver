"""Compare generated MOOSE expressions against the actual C++ solver (no MOOSE required)."""
import importlib.util
import math
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('generator', ROOT / 'moose/generate_kinetics1d.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def main():
    n = 7
    temperatures = [1736.6, 300., 2000., 1736.60001, 1700., 2500., 1920.]
    state = {}
    for i in range(n):
        y1, y2 = .7-.05*i, .1+.01*i
        for name, value in [('Y1', y1), ('Y2', y2), ('Y3', 1-y1-y2), ('T', temperatures[i])]:
            state[f'{name}_{i}'] = value
    equations = generator.equations(n, 250.)
    env = dict(generator.constants(), exp=math.exp, choose=lambda condition, a, b: a if condition else b)

    def rhs(values):
        scope = dict(env, **values)
        return {key: -eval(expression.replace('if(', 'choose('), {'__builtins__': {}}, scope)
                for key, (expression, _) in equations.items()}

    original = dict(state)
    reference_rhs = rhs(state)
    for _ in range(20):
        r = rhs(state)
        s1 = {k: state[k]+1e-5*r[k] for k in state}
        r = rhs(s1)
        s2 = {k: .75*state[k]+.25*s1[k]+.25*1e-5*r[k] for k in state}
        r = rhs(s2)
        state = {k: state[k]/3+2*s2[k]/3+2*1e-5*r[k]/3 for k in state}
    source = '''#define main kinetics_original_main
#include "''' + str(ROOT / 'src/kinetics1D.C') + '''"
#undef main
#include <iomanip>
int main() {
 params::N=7;params::l=250;params::dt=1e-5;params::T_init=1920;params::T_0=300;
 params::update();initializeState();
'''
    for i in range(n):
        for k, field in enumerate(('Y1', 'Y2', 'Y3', 'T')):
            source += f' state[{k}][{i}]={original[f"{field}_{i}"]:.17g};\n'
    source += '''std::cout<<std::setprecision(17);
 auto r=assembleResidual(state);
 const char* fields[]={"Y1","Y2","Y3","T"};
 for(int k=0;k<4;++k)for(int i=0;i<7;++i)std::cout<<"R "<<fields[k]<<'_'<<i<<' '<<r[k][i]<<'\\n';
 for(int step=0;step<20;++step)RK3Step(state);
 for(int k=0;k<4;++k)for(int i=0;i<7;++i)std::cout<<"S "<<fields[k]<<'_'<<i<<' '<<state[k][i]<<'\\n';
}
'''
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        (path / 'reference.C').write_text(source)
        subprocess.run(['c++', '-std=c++17', '-O2', '-I'+str(ROOT / 'include'),
                        '-I/opt/homebrew/include/eigen3', str(path / 'reference.C'),
                        '-o', str(path / 'reference')], check=True)
        output = subprocess.check_output([str(path / 'reference')], text=True)
    for line in output.splitlines():
        kind, key, value = line.split()
        expected = reference_rhs[key] if kind == 'R' else state[key]
        assert math.isclose(float(value), expected, rel_tol=2e-12, abs_tol=1e-10), (line, expected)
    # Preserve C++ integer-index center and strict left-width comparisons.
    c = generator.constants()
    centered = generator.initial_values(11, 10, 1920, 300, 0, c)
    assert [i for i in range(11) if centered[f'T_{i}'] == 1920] == [4, 5]
    left = generator.initial_values(11, 10, 1920, 300, 2, c)
    assert [i for i in range(11) if left[f'T_{i}'] == 1920] == [0, 1]
    assert (ROOT / 'moose/kinetics1D.i').read_text() == generator.render()
    print('Passed: generated residuals and 20 SSP RK3 steps match kinetics1D.C; both heat branches, endpoints, ICs and input regeneration checked.')


if __name__ == '__main__':
    main()
