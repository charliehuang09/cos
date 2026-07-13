#include "control_loop/control_loop.h"
namespace control_loop {

class INode {
 public:
  INode(const INode&) = default;
  virtual ~INode() = default;
  virtual void MaybeRun(Context) = 0;
};

}  // namespace control_loop
