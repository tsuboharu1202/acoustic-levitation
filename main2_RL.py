import BallPopController_RL
import BallPopRenderer
controller = BallPopController_RL.BallPopController()
controller.RunMainLoop()
renderer = BallPopRenderer.BallPopRenderer(controller)
controller.Closer()