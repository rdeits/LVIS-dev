__precompile__()

module LearningMPC

using LCPSim: LCPUpdate, contact_force 
using DrakeVisualizer: PolyLine, Visualizer, ArrowHead, settransform!, setgeometry!
using RigidBodyDynamics: transform_to_root, MechanismState, set_configuration!, configuration

export playback

function playback(vis::Visualizer, results::AbstractVector{<:LCPUpdate}, Δt = 0.01)
    state = MechanismState{Float64}(results[1].state.mechanism)
    for result in results
        set_configuration!(state, configuration(result.state))
        settransform!(vis, state)
        for (body, contacts) in result.contacts
            for (i, contact) in enumerate(contacts)
                f = contact_force(contact)
                p = transform_to_root(state, contact.point.frame) * contact.point
                v = vis[:forces][Symbol(body)][Symbol(i)]
                setgeometry!(v, PolyLine([p.v, (p + 0.1*f).v]; end_head=ArrowHead()))
            end
        end
        sleep(Δt)
    end
end

end
