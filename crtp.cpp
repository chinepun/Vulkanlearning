#include <iostream>
using namespace std;

template <typename Child>
struct Base
{
    void interface()

    {
        static_cast<Child*>(this)->implementation();
    }
    void implementation(){std::cout << "Base Implementation";}
};

struct Derived : Base<Derived>
{
    void implementation()
    {
        cerr << "Derived implementation\n";
    }
};

template<typename deduced>
struct  Cont
{
    Cont(deduced *c){}
};


int main()
{
    Derived d;
    Base<Derived> b;
    b.interface();
    d.interface();  // Prints "Derived implementation"

// class Template Argument Deduction(e.g std::vector specifyTypeinConstruction = {1,2,3};)
    Cont a1((int*)2);
    int x =  123;
    Cont<int> a2(&x);
}