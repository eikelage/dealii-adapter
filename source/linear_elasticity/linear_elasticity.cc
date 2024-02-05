#include "include/linear_elasticity.h"

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/table_handler.h>

#include <adapter/adapter.h>
#include <adapter/parameters.h>
#include <adapter/time_handler.h>

#include <fstream>
#include <iostream>

#include "include/postprocessor.h"

// The Linear_Elasticity case includes a linear elastic material with a one-step
// theta time integration
namespace Linear_Elasticity
{
  using namespace dealii;

  // Constructor
  template <int dim>
  ElastoDynamics<dim>::ElastoDynamics(const std::string &parameter_file)
    : parameters(parameter_file)
    , interface_boundary_id(6)
    , dof_handler(triangulation)
    , fe(FE_Q<dim>(parameters.poly_degree), dim)
    , mapping(MappingQGeneric<dim>(parameters.poly_degree))
    , quad_order(parameters.poly_degree + 1)
    , body_force_enabled(parameters.body_force.norm() > 1e-15)
    , timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
    , time(parameters.end_time, parameters.delta_t)
    , adapter(parameters, interface_boundary_id)
  {std::cout << "-------------------------------- " << parameter_file << std::endl;}



  // Destructor
  template <int dim>
  ElastoDynamics<dim>::~ElastoDynamics()
  {
    dof_handler.clear();
  }

  template <int dim>
  void
  ElastoDynamics<dim>::make_grid()
  {
    uint n_x, n_y;

    // Both preconfigured cases consist of a rectangle
    Point<dim> point_bottom;
    Point<dim> point_tip;
    Point<dim> point_mid_left;
    Point<dim> point_mid_right;

    // boundary IDs are obtained through colorize = true
    uint id_jf_left, id_jf_right, id_jf_bottom, id_jf_top; //, id_flap_out_of_plane_bottom, id_flap_out_of_plane_top;

    if (parameters.scenario == "JF"){
        n_x = 3;
        n_y = 61;

        clamped_mesh_id              = 0;

        point_bottom   = dim == 3 ? Point<dim>(124.0, 120.0, 0.0) * 1E-3 :
                                    Point<dim>(124.0, 120.0) * 1E-3;                    
        point_tip      = dim == 3 ? Point<dim>(126.0, 240.0, 1.0) * 1E-3 :
                                    Point<dim>(126.0, 240.0) * 1E-3;

        id_jf_left   = 0; // x direction
        id_jf_right  = 1;
        id_jf_bottom = 2; // y direction
        id_jf_top    = 3;

        const std::vector<unsigned int> repetitions = std::vector<unsigned int>({n_x, n_y});

        GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, point_bottom, point_tip, true);

         // Refine all cells global_refinement times
        const unsigned int global_refinement = 0;
        triangulation.refine_global(global_refinement);

        // Iterate over all cells and set the IDs
        for (const auto &cell : triangulation.active_cell_iterators())
          for (const auto &face : cell->face_iterators()){
            if (face->at_boundary() == true){
                // Boundaries for the interface
                if (face->boundary_id() == id_jf_left || face->boundary_id() == id_jf_right || face->boundary_id() == id_jf_bottom)
                  face->set_boundary_id(interface_boundary_id);
                // Boundaries clamped in all directions
                else if (face->boundary_id() == id_jf_top)
                  face->set_boundary_id(interface_boundary_id);
              }
          }
    }
    else if (parameters.scenario ==  "OBJ"){
        n_x = 20;
        n_y = 20;

        point_bottom = dim == 3 ? Point<dim>(171, 176, 0) * 1E-3 :
                                  Point<dim>(171, 176)* 1E-3;
        point_tip    = dim == 3 ? Point<dim>(179, 184, 1) * 1E-3 :
                                  Point<dim>(179, 184) * 1E-3; // flap has a 0.1 width
        
        const std::vector<unsigned int> repetitions = std::vector<unsigned int>({n_x, n_y});

        GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              repetitions,
                                              point_bottom,
                                              point_tip,
                                              false);

        // Refine all cells global_refinement times
        const unsigned int global_refinement = 0;
        triangulation.refine_global(global_refinement);

        // Iterate over all cells and set the IDs
        for (const auto &cell : triangulation.active_cell_iterators())
          for (const auto &face : cell->face_iterators())
            if (face->at_boundary() == true)
              face->set_boundary_id(interface_boundary_id);
    }
    else {
      std::cout << "Scenario Undefined" << std::endl;
    }
  }



  template <int dim>
  void
  ElastoDynamics<dim>::setup_system()
  {
    // This follows the usual dealii steps
    dof_handler.distribute_dofs(fe);
    hanging_node_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            hanging_node_constraints);
    hanging_node_constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    hanging_node_constraints,
                                    /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from(dsp);

    // Initialize relevant matrices
    mass_matrix.reinit(sparsity_pattern);
    stiffness_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);
    stepping_matrix.reinit(sparsity_pattern);

    // Initialize all vectors
    old_velocity.reinit(dof_handler.n_dofs());
    velocity.reinit(dof_handler.n_dofs());

    old_displacement.reinit(dof_handler.n_dofs());
    displacement.reinit(dof_handler.n_dofs());

    system_rhs.reinit(dof_handler.n_dofs());
    old_stress.reinit(dof_handler.n_dofs());
    stress.reinit(dof_handler.n_dofs());

    body_force_vector.reinit(dof_handler.n_dofs());

    std::cout.imbue(std::locale(""));
    std::cout << "Triangulation:"
              << "\n\t Scenario: " << parameters.scenario
              << "\n\t Number of active cells: "
              << triangulation.n_active_cells()
              << "\n\t Polynomial degree: " << parameters.poly_degree
              << "\n\t Number of degrees of freedom: " << dof_handler.n_dofs()
              << "\n\t ShearModulus: " << parameters.mu
              << std::endl;

    // Define alias for time dependent variables as described above
    state_variables = {
      &old_velocity, &velocity, &old_displacement, &displacement, &old_stress};

    // loads at time 0
    // TODO: Check, if initial conditions should be set at the beginning
    old_stress = 0.0;
  }




  template <int dim>
  void
  ElastoDynamics<dim>::assemble_system()
  {
    QGauss<dim> quadrature_formula(quad_order);

    FEValues<dim> fe_values(mapping,
                            fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> lambda_values(n_q_points);
    std::vector<double> mu_values(n_q_points);

    // Lame constants
    Functions::ConstantFunction<dim> lambda(parameters.lambda),
      mu(parameters.mu);

    // Assemble the stiffness matrix according to a linear material law using
    // the lame paramters
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;

        fe_values.reinit(cell);

        // Next we get the values of the coefficients at the quadrature
        // points.
        lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
        mu.value_list(fe_values.get_quadrature_points(), mu_values);
        // externalForce(fe_values.get_quadrature_points(), rhs_values);

        // Then assemble the entries of the local stiffness matrix
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const unsigned int component_i =
              fe.system_to_component_index(i).first;

            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                const unsigned int component_j =
                  fe.system_to_component_index(j).first;

                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                  {
                    cell_matrix(i, j) +=
                      // the first term is (lambda d_i u_i, d_j v_j) + (mu d_i
                      // u_j, d_j v_i).
                      (                                                  //
                        (fe_values.shape_grad(i, q_point)[component_i] * //
                         fe_values.shape_grad(j, q_point)[component_j] * //
                         lambda_values[q_point])                         //
                        +                                                //
                        (fe_values.shape_grad(i, q_point)[component_j] * //
                         fe_values.shape_grad(j, q_point)[component_i] * //
                         mu_values[q_point])                             //
                        +                                                //
                        // the second term is (mu nabla u_i, nabla v_j).
                        ((component_i == component_j) ?        //
                           (fe_values.shape_grad(i, q_point) * //
                            fe_values.shape_grad(j, q_point) * //
                            mu_values[q_point]) :              //
                           0)                                  //
                        ) *                                    //
                      fe_values.JxW(q_point);                  //
                  }
              }
          }

        // The transfer from local degrees of freedom into the global matrix
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i){
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              stiffness_matrix.add(local_dof_indices[i],
                                   local_dof_indices[j],
                                   cell_matrix(i, j));
          }
    
      }
    // Here, we use the MatrixCreator to create a mass matrix, which is constant
    // through the whole simulation
    {
      Functions::ConstantFunction<dim> rho_f(parameters.rho);

      MatrixCreator::create_mass_matrix(
        mapping, dof_handler, QGauss<dim>(quad_order), mass_matrix, &rho_f);
    }

    // Then, we save the system_matrix, which is needed every timestep
    stepping_matrix.copy_from(stiffness_matrix);

    stepping_matrix *= time.get_delta_t() * time.get_delta_t() *
                       parameters.theta * parameters.theta;

    stepping_matrix.add(1, mass_matrix);

    hanging_node_constraints.condense(stepping_matrix);



    // Calculate contribution of gravity and store them in gravitational_force
    if (1 || body_force_enabled){
        Vector<double> bf_vector(dim);
        for (uint d = 0; d < dim; ++d)
          bf_vector[d] = parameters.rho * parameters.body_force[d];

        // Create a constant function object
        Functions::ConstantFunction<dim> bf_function(bf_vector);

        if(parameters.scenario == "JF"){
          std::cout<< "==================== CHECK: Attachment Points Top ========================" << std::endl;
          auto attachment_top_cell(GridTools::find_active_cell_around_point(mapping, dof_handler, Point<dim>(126.0,240.0) * 1E-3, std::vector<bool>(), 1E-1 ));
          std::cout<< "==================== CHECK: Attachment Points Bot ========================" << std::endl;
          auto attachment_bot_cell(GridTools::find_active_cell_around_point(mapping, dof_handler, Point<dim>(126.0,120.0) * 1E-3, std::vector<bool>(), 1E-1 ));

          clamped_dof_idx.reserve(3*dofs_per_cell);
          int n_x = 3;
          double thickness = 2.0;

          // for(int i(0); i != n_x; ++i){
          //   std::vector<types::global_dof_index> tmp(dofs_per_cell);
          //   auto clamped_cells(GridTools::find_active_cell_around_point(mapping, dof_handler, Point<dim>((124.0+i*thickness/n_x),180.0) * 1E-3, std::vector<bool>(), 1E-1 ));
          //   clamped_cells.first->get_dof_indices(tmp);
          //   clamped_dof_idx.insert(std::end(clamped_dof_idx), std::begin(tmp), std::end(tmp));
          //   std::cout<< "==================== CHECK: Clamped Cell Idx ========================" << std::endl;          
          //   for(int idx: clamped_dof_idx){
          //     std::cout << idx << std::endl;
          //   }
          // }

          for(int i(0); i != n_x; ++i){
            std::vector<types::global_dof_index> tmp(dofs_per_cell);
            auto clamped_cells(GridTools::find_active_cell_around_point(mapping, dof_handler, Point<dim>((124.0+i*thickness/n_x),180.0) * 1E-3, std::vector<bool>(), 1E-1 ));
            clamped_cells.first->get_dof_indices(tmp);
            for (const auto &face : clamped_cells.first->face_iterators())
              if (face->at_boundary() == true)
                face->set_boundary_id(clamped_mesh_id);
          }

          std::cout<< "==================== CHECK: Attachement Point Idx ========================" << std::endl;
          attchmntpnt_top_dof_idx.resize(dofs_per_cell);
          attchmntpnt_bot_dof_idx.resize(dofs_per_cell);
          std::cout<< "-------------------- CHECK: Attachement Point Idx Top --------------------" << std::endl;
          attachment_top_cell.first->get_dof_indices(attchmntpnt_top_dof_idx);
          std::cout<< "-------------------- CHECK: Attachement Point Idx Bot --------------------" << std::endl;
          attachment_bot_cell.first->get_dof_indices(attchmntpnt_bot_dof_idx);
          std::cout<< "==================== CHECK: assemble_system complete ========================" << std::endl;
        }
      }
  }


  // Process RHS assembly, which is the coupling data (stress) in this case
  template <int dim>
  void
  ElastoDynamics<dim>::assemble_rhs(){
    std::cout<< "==================== CHECK: assemble_rhs start ========================" << std::endl;
    timer.enter_subsection("Assemble rhs");
    // In case we get consistent data
    if (parameters.data_consistent)
      assemble_consistent_loading();
    else // In case we get conservative data
      system_rhs = stress;
    // Update time dependent variables related to the previous time step t_n
    old_velocity     = velocity;
    old_displacement = displacement;

    auto tmp_body_force_vector(body_force_vector);
    if(parameters.scenario == "JF"){
      double force_mag(0.0);
      attachment_point_top(0) = 126.0 * 1E-3;
      attachment_point_top(1) = 240.0 * 1E-3;
      attachment_point_bot(0) = 126.0 * 1E-3;
      attachment_point_bot(1) = 120.0 * 1E-3;
      Point<dim> dsplcmnt_top(0.0,0.0);
      Point<dim> dsplcmnt_bot(0.0,0.0);
      Point<dim> origin(126.0 * 1E-3,180.0 * 1E-3);
      Point<dim> tmp;

      for (unsigned i(0); i != fe.dofs_per_cell; ++i){
        const unsigned int component_i = fe.system_to_component_index(i).first;
        dsplcmnt_top(component_i) += displacement[attchmntpnt_top_dof_idx[i]];
        dsplcmnt_bot(component_i) += displacement[attchmntpnt_bot_dof_idx[i]];
      }

      dsplcmnt_top = dsplcmnt_top/(fe.dofs_per_cell/2.0);
      dsplcmnt_bot = dsplcmnt_bot/(fe.dofs_per_cell/2.0);

      std::cout << "------------ Displacement Top: (" << dsplcmnt_top(0) << ", " << dsplcmnt_top(1) << ")" << std::endl;

      attachment_point_top = attachment_point_top + dsplcmnt_top;
      attachment_point_bot = attachment_point_bot + dsplcmnt_bot;
      force_mag = parameters.f_mag;
      
      if(time.current() < 0.1 && 0){
        // force_mag = 0.1;
        force_top(0) = force_mag;
        force_top(1) = 0.0;
        force_bot(0) = force_mag;
        force_bot(1) = 0.0;
      }
      else if (time.current() > 0.0 && time.current() < 0.8){
        force_mag = time.current()/0.8 * parameters.f_mag;
        force_top = force_mag*((origin - attachment_point_top)/abs(origin.distance(attachment_point_top)));
        tmp = force_top;
        force_top(0) = -tmp(1);
        force_top(1) = tmp(0);
        force_bot = force_mag*((origin - attachment_point_bot)/abs(origin.distance(attachment_point_bot)));
        tmp = force_bot;
        force_bot(0) = tmp(1);
        force_bot(1) = -tmp(0);
      }
      else{
        force_mag = (1-(time.current()-0.8)/0.2) * parameters.f_mag;
        force_top(0) = 0;
        force_top(1) = 0.0;
        force_bot(0) = 0;
        force_bot(1) = 0.0;
      }

      // force_top = force_mag*((origin - attachment_point_top)/abs(origin.distance(attachment_point_top)));
      // tmp = force_top;
      // force_top(0) = -tmp(1);
      // force_top(1) = tmp(0);
      // force_bot = force_mag*((origin - attachment_point_bot)/abs(origin.distance(attachment_point_bot)));
      // tmp = force_bot;
      // force_bot(0) = tmp(1);
      // force_bot(1) = -tmp(0);
      
      for (unsigned i(0); i != fe.dofs_per_cell; ++i){
        const unsigned int component_i = fe.system_to_component_index(i).first;
        tmp_body_force_vector[attchmntpnt_top_dof_idx[i]] += force_top(component_i); 
        tmp_body_force_vector[attchmntpnt_bot_dof_idx[i]] += force_bot(component_i); 
      }
    }
    
    system_rhs += tmp_body_force_vector;
    std::cout<< "==================== CHECK: Force Application Done ========================" << std::endl;

    // Assemble global RHS:
    // RHS=(M-theta*(1-theta)*delta_t^2*K)*V_n - delta_t*K* D_n +
    // delta_t*theta*F_n+1 + delta_t*(1-theta)*F_n

    // tmp vector to store intermediate results
    Vector<double> tmp;
    tmp.reinit(dof_handler.n_dofs());

    tmp = system_rhs;

    system_rhs *= time.get_delta_t() * parameters.theta; 
    system_rhs.add(time.get_delta_t() * (1 - parameters.theta), old_stress);
    old_stress = tmp;

    mass_matrix.vmult(tmp, old_velocity);
    system_rhs.add(1, tmp);

    stiffness_matrix.vmult(tmp, old_velocity);
    system_rhs.add(-parameters.theta * time.get_delta_t() * time.get_delta_t() *
                     (1 - parameters.theta),
                   tmp);

    stiffness_matrix.vmult(tmp, old_displacement);
    system_rhs.add(-time.get_delta_t(), tmp);
    // certain rows and columns
    system_matrix = 0.0;
    system_matrix.copy_from(stepping_matrix);
  
    // Set Dirichlet BCs:
    // clamped in all directions
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             clamped_mesh_id,
                                             Functions::ZeroFunction<dim>(dim),
                                             boundary_values);
    
    std::cout<< "==================== CHECK: Applying BC ========================" << std::endl;
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       velocity,
                                       system_rhs);
    

    timer.leave_subsection("Assemble rhs");
  }


  // Process RHS assembly, which is the coupling data (stress) in this case
  template <int dim>
  void
  ElastoDynamics<dim>::assemble_consistent_loading()
  { // Initialize all objects as usual
    system_rhs = 0.0;

    // Quadrature formula for integration over faces (dim-1)
    QGauss<dim - 1> face_quadrature_formula(quad_order);

    FEFaceValues<dim> fe_face_values(mapping,
                                     fe,
                                     face_quadrature_formula,
                                     update_values | update_JxW_values);

    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    Vector<double>                       cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


    // In order to get the local fe values
    std::vector<Vector<double>> local_stress(n_face_q_points,
                                             Vector<double>(dim));

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_rhs = 0;

        // Assemble the right-hand side force vector each timestep
        // by applying contributions only on the coupling interface
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary() == true &&
              face->boundary_id() == interface_boundary_id)
            {
              fe_face_values.reinit(cell, face);
              // Extract relevant data from the global stress vector by using
              // 'get_function_values()'
              // In contrast to the nonlinear solver, no pull back is performed.
              // The equilibrium is stated in reference configuration, but only
              // valid for very small deformations
              fe_face_values.get_function_values(stress, local_stress);

              for (unsigned int f_q_point = 0; f_q_point < n_face_q_points;
                   ++f_q_point)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    const unsigned int component_i =
                      fe.system_to_component_index(i).first;

                    cell_rhs(i) += fe_face_values.shape_value(i, f_q_point) *
                                   local_stress[f_q_point][component_i] *
                                   fe_face_values.JxW(f_q_point);
                  }
            }

        // Local dofs to global
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
          }
      }
  }



  template <int dim>
  void
  ElastoDynamics<dim>::solve()
  {
    timer.enter_subsection("Solve system");

    uint   lin_it  = 1;
    double lin_res = 0.0;

    // Solve the linear system either using an iterative CG solver with SSOR or
    // a direct solver using UMFPACK
    if (parameters.type_lin == "CG")
      {
        std::cout << "\t CG solver: " << std::endl;

        const int solver_its =
          system_matrix.m() * parameters.max_iterations_lin;
        const double tol_sol = 1.e-10;

        SolverControl         solver_control(solver_its, tol_sol);
        GrowingVectorMemory<> GVM;
        SolverCG<>            solver_CG(solver_control, GVM);

        PreconditionSSOR<> preconditioner;
        preconditioner.initialize(system_matrix, 1.2);

        solver_CG.solve(system_matrix, velocity, system_rhs, preconditioner);

        lin_it  = solver_control.last_step();
        lin_res = solver_control.last_value();
      }
    else if (parameters.type_lin == "Direct")
      {
        std::cout << "\t Direct solver: " << std::endl;

        SparseDirectUMFPACK A_direct;
        A_direct.initialize(system_matrix);
        A_direct.vmult(velocity, system_rhs);
      }
    else
      Assert(parameters.type_lin == "Direct" || parameters.type_lin == "CG",
             ExcNotImplemented());

    // assert divergence
    Assert(velocity.linfty_norm() < 1e4, ExcMessage("Linear system diverged"));
    std::cout << "\t     No of iterations:\t" << lin_it
              << "\n \t     Final residual:\t" << lin_res << std::endl;
    hanging_node_constraints.distribute(velocity);

    timer.leave_subsection("Solve system");
  }



  template <int dim>
  void
  ElastoDynamics<dim>::update_displacement()
  {
    // D_n+1= D_n + delta_t*theta* V_n+1 + delta_t*(1-theta)* V_n
    displacement.add(time.get_delta_t() * parameters.theta, velocity);
    displacement.add(time.get_delta_t() * (1 - parameters.theta), old_velocity);
  }



  template <int dim>
  void
  ElastoDynamics<dim>::output_results() const
  {
    timer.enter_subsection("Output results");
    DataOut<dim> data_out;

    // Note: There is at least paraView v 5.5 needed to visualize this output
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);

    // The postprocessor class computes straines and passes the displacement to
    // the output
    Postprocessor<dim> postprocessor;
    data_out.add_data_vector(displacement, postprocessor);

    TableHandler table_other_data;
    table_other_data.add_value("ax_top", attachment_point_top(0));
    table_other_data.add_value("ay_top", attachment_point_top(1));
    table_other_data.add_value("az_top", 0.0);
    table_other_data.add_value("fx_top", force_top(0));
    table_other_data.add_value("fy_top", force_top(1));
    table_other_data.add_value("fz_top", 0.0);

    table_other_data.add_value("ax_bot", attachment_point_bot(0));
    table_other_data.add_value("ay_bot", attachment_point_bot(1));
    table_other_data.add_value("az_bot", 0.0);
    table_other_data.add_value("fx_bot", force_bot(0));
    table_other_data.add_value("fy_bot", force_bot(1));
    table_other_data.add_value("fz_bot", 0.0);

    //std::string path="/home/elage/repos/FSI-sim2real/PreCice/perpendicular-flap/solid-dealii/custom_data/data_" + std::to_string(time.current()) + ".csv"; // save by sim times
    std::string path;
    if(parameters.scenario == "OBJ")
      path="/home/elage/repos/FSI-sim2real/PreCice/multi_body_jelly_fish/simFiles/solid-obj-dealii/custom_data/data_" + std::to_string(time.get_timestep()/parameters.output_interval) + ".csv"; // save by iteration
    else
      path="/home/elage/repos/FSI-sim2real/PreCice/multi_body_jelly_fish/simFiles/solid-jf-dealii/custom_data/data_" + std::to_string(time.get_timestep()/parameters.output_interval) + ".csv"; // save by iteration

    std::ofstream out_file(path);
    table_other_data.write_text(out_file, TableHandler::TextOutputFormat::org_mode_table);
    out_file.close();

    // visualize the displacements on a displaced grid
    MappingQEulerian<dim> q_mapping(parameters.poly_degree,
                                    dof_handler,
                                    displacement);
    data_out.build_patches(q_mapping,
                           parameters.poly_degree,
                           DataOut<dim>::curved_boundary);

    std::ofstream output(
      parameters.output_folder + "/solution-" +
      Utilities::int_to_string(time.get_timestep() / parameters.output_interval,
                               3) +
      ".vtk");
    data_out.write_vtk(output);
    std::cout << "\t Output written to solution-" +
                   Utilities::int_to_string(time.get_timestep() /
                                              parameters.output_interval,
                                            3) +
                   ".vtk \n"
              << std::endl;
    timer.leave_subsection("Output results");
  }



  template <int dim>
  void
  ElastoDynamics<dim>::run()
  {
    // In the beginning, we create the mesh and set up the data structures
    make_grid();
    setup_system();
    output_results();
    assemble_system();

    // Then, we initialize preCICE i.e. we pass our mesh and coupling
    // information to preCICE
    adapter.initialize(dof_handler, displacement, stress);

    // Then, we start the time loop. The loop itself is steered by preCICE. This
    // line replaces the usual 'while( time < end_time)'
    while (adapter.precice.isCouplingOngoing())
      {
        // In case of an implicit coupling, we need to store time dependent
        // data, in order to reload it later. The decision, whether it is
        // necessary to store the data is handled by preCICE as well
        adapter.save_current_state_if_required(state_variables, time);

        // Afterwards, we start the actual time step computation
        time.increment();

        std::cout << std::endl
                  << "Timestep " << time.get_timestep() << " @ " << std::fixed
                  << time.current() << "s" << std::endl;

        // Assemble the time dependent contribution obtained from the Fluid
        // participant
        assemble_rhs();

        // ...and solver the system
        solve();

        // Update time dependent data according to the theta-scheme
        update_displacement();

        // Then, we exchange data with other participants. Most of the work is
        // done in the adapter: We just need to pass both data vectors with
        // coupling data to the adapter. In case of FSI, 'displacement' is the
        // data we calculate and pass to preCICE and 'stress' is the (global)
        // vector filled by preCICE/ the Fluid participant.
        // Depending on the coupling scheme, we need to wait here for other
        // participant to finish their time step. Therefore, we measure the
        // timings around this functionality
        timer.enter_subsection("Advance adapter");
        adapter.advance(displacement, stress, time.get_delta_t());
        timer.leave_subsection("Advance adapter");

        // Next, we reload the data we have previosuly stored in the beginning
        // of the time loop. This is only relevant for implicit couplings and
        // preCICE steeres the reloading depending on the specific
        // configuration.
        adapter.reload_old_state_if_required(state_variables, time);

        // At last, we ask preCICE, whether this coupling time step (= time
        // window in preCICE terms) is finished and write the result files
        if (adapter.precice.isTimeWindowComplete() &&
            time.get_timestep() % parameters.output_interval == 0)
          output_results();
      }

    // After the time loop, we finalize the coupling i.e. terminate
    // communication etc.
    adapter.precice.finalize();
  }

  template class ElastoDynamics<DIM>;
} // namespace Linear_Elasticity
