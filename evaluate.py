import glob
import os
import os.path as osp
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from utils.eval_utils import *

@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters."""
    models_dir: str
    gt_dir: str
    output_dir: str
    extension: str = ".png"
    enable_depth: bool = True
    gpu_id: int = 0  # GPU device ID to use, default is 0
    
    setups: List[str] = None
    views: List[int] = None
    num_view_list: List[int] = None
    num_images_list: List[int] = None
    metrics: List[Metrics] = None
    
    def __post_init__(self):
        if self.setups is None:
            self.setups = ["basketball", "bottles", "coffee", "chikawa", "color", "projector", "wukong"]
        if self.views is None:
            self.views = [1, 6, 11, 16, 21, 26, 27, 28, 29, 30, 31, 32, 33]
        if self.num_view_list is None:
            self.num_view_list = [25]
        if self.num_images_list is None:
            self.num_images_list = []
        if self.metrics is None:
            self.metrics = [Metrics.PSNR, Metrics.SSIM, Metrics.LPIPS]
            
        # Define trained and novel viewpoints
        self.trained_views = [1, 6, 11, 16, 21]
        self.novel_views = [26, 27, 28, 29, 30, 31, 32, 33]


class ModelEvaluator:
    """Main evaluator class for model performance assessment."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
        # Use the GPU device that was set globally with CUDA_VISIBLE_DEVICES
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')  # After CUDA_VISIBLE_DEVICES, it's always 0
            print(f"Using device: {self.device}")
        else:
            self.device = torch.device('cpu')
            print("Using device: CPU")
            
        os.makedirs(config.output_dir, exist_ok=True)
        
    def _get_paths(self, setup: str, view_id: int, num_view: int = None, num_images: int = None) -> Dict[str, str]:
        """Generate all necessary paths for evaluation."""
        paths = {}
        
        # Determine which path format to use
        if num_view is not None:
            # Use num_view format
            paths['pred_relit'] = osp.join(
                self.config.models_dir, "setups", setup, "num_view", 
                str(num_view), "render", str(view_id).zfill(2), "relit"
            )
            paths['pred_depth'] = osp.join(
                self.config.models_dir, "setups", setup, "num_view",
                str(num_view), "render", str(view_id).zfill(2), "scene", "depth.tiff"
            ) if self.config.enable_depth else None
        elif num_images is not None:
            # Use num_images format
            paths['pred_relit'] = osp.join(
                self.config.models_dir, "setups", setup, "num_images", 
                str(num_images), "render", str(view_id).zfill(2), "relit"
            )
            paths['pred_depth'] = osp.join(
                self.config.models_dir, "setups", setup, "num_images",
                str(num_images), "render", str(view_id).zfill(2), "scene", "depth.tiff"
            ) if self.config.enable_depth else None
        else:
            raise ValueError("Either num_view or num_images must be provided")
        
        # Ground truth paths (same for both formats)
        view_dir = osp.join(self.config.gt_dir, "setups", setup, "views", str(view_id).zfill(2))
        paths['view_dir'] = view_dir
        paths['gt_relit'] = osp.join(view_dir, "cam", "raw", "test")
        paths['cam_mask'] = osp.join(view_dir, "cam", "raw", "mask", "mask.png")
        
        if self.config.enable_depth:
            paths['gt_depth'] = osp.join(view_dir, "recon", "depthGT_cleaned.txt")
            paths['params'] = osp.join(view_dir, "params", "params.yml")
            
        return paths
    
    def _validate_paths(self, paths: Dict[str, str]) -> bool:
        """Validate that all required paths exist."""
        required_files = ['view_dir', 'cam_mask']
        if self.config.enable_depth and paths.get('pred_depth'):
            required_files.extend(['pred_depth', 'gt_depth', 'params'])
            
        for key in required_files:
            if not osp.exists(paths[key]):
                print(f"Missing required path: {paths[key]}")
                return False
        return True
    
    def _load_images(self, image_dir: str) -> Optional[torch.Tensor]:
        """Load all images from a directory as a tensor stack."""
        image_paths = sorted(glob.glob(osp.join(image_dir, f"*{self.config.extension}")))
        if not image_paths:
            print(f"No images found in {image_dir}")
            return None
            
        images = []
        for path in image_paths:
            img = load_image_as_tensor(path, self.device)
            if img is not None:
                images.append(img)
                
        return torch.stack(images) if images else None
    
    def _evaluate_single_case(self, setup: str, view_id: int, num_view: int = None, num_images: int = None) -> Dict:
        """Evaluate a single case configuration."""
        paths = self._get_paths(setup, view_id, num_view=num_view, num_images=num_images)
        
        if not self._validate_paths(paths):
            return {}
            
        # Load images
        gt_images = self._load_images(paths['gt_relit'])
        pred_images = self._load_images(paths['pred_relit'])
        
        if gt_images is None or pred_images is None:
            case_id = f"{setup}-{view_id}-{num_view if num_view else num_images}"
            print(f"Failed to load images for {case_id}")
            return {}
            
        if len(gt_images) != len(pred_images):
            print(f"Image count mismatch: {len(gt_images)} vs {len(pred_images)}")
            return {}
        
        results = {}
        
        # Depth evaluation
        if self.config.enable_depth and paths.get('pred_depth'):
            try:
                cam_KRT = loadCalib(paths['params'])["camK"][0, :3, :3]
                depth_sl = load_depth_data(paths['gt_depth'])
                pc_sl = depth_to_points(depth_sl, cam_KRT)
                sl_mask = depth_sl > 0
                
                sl_mask_tensor = torch.tensor(sl_mask, device=self.device).unsqueeze(0).expand_as(pred_images)
                
                # Compute metrics with SL mask
                masked_results = computeMetrics(
                    pred_images * sl_mask_tensor, 
                    gt_images * sl_mask_tensor, 
                    self.config.metrics
                )
                
                results["sl_mask"] = {metric.value: float(value) for metric, value in masked_results.items()}
                
                # Depth error
                depth_pred = load_depth_data(paths['pred_depth'])
                pc_pred = depth_to_points(depth_pred, cam_KRT)
                d_err = align_point_clouds(pc_pred, pc_sl, mask=sl_mask)
                results["sl_mask"]["d_err"] = d_err
                
                case_id = f"Setup: {setup}, View: {view_id}"
                if num_view:
                    case_id += f", Train: {num_view}"
                if num_images:
                    case_id += f", Images: {num_images}"
                print(case_id)
                for metric, value in results["sl_mask"].items():
                    print(f"  {metric}: {value:.4f}")
                    
            except Exception as e:
                print(f"Depth evaluation failed: {e}")
                
        return results
    
    def _validate_all_paths(self) -> bool:
        """Check all paths before evaluation and list missing ones."""
        print("Checking paths...")
        missing_paths = []
        
        for setup in self.config.setups:
            for view_id in self.config.views:
                # Check num_view paths
                for num_view in self.config.num_view_list:
                    if num_view == 25 or view_id in self.config.novel_views:
                        paths = self._get_paths(setup, view_id, num_view=num_view)
                        for key, path in paths.items():
                            if path and not osp.exists(path):
                                missing_paths.append(path)
                
                # Check num_images paths
                for num_images in self.config.num_images_list:
                    paths = self._get_paths(setup, view_id, num_images=num_images)
                    for key, path in paths.items():
                        if path and not osp.exists(path):
                            missing_paths.append(path)
        
        if missing_paths:
            print("Missing paths:")
            for path in sorted(set(missing_paths)):
                print(path)
            return False
        
        print("All paths exist.")
        return True
    
    def run_evaluation(self) -> Dict:
        """Run complete evaluation across all configurations."""
        # Validate all paths before starting
        if not self._validate_all_paths():
            raise FileNotFoundError("Some required paths are missing. Please check the logs above.")
            
        if not self._confirm_start():
            return {}
            
        results = {}
        
        # Calculate total cases considering the filtering for num_view
        total_cases = 0
        for setup in self.config.setups:
            for view_id in self.config.views:
                # Count num_view cases
                for num_view in self.config.num_view_list:
                    # For num_view, only count novel views unless num_view=25
                    if num_view == 25 or view_id in self.config.novel_views:
                        total_cases += 1
                
                # Count num_images cases (always all views)
                total_cases += len(self.config.num_images_list)
        
        current_case = 0
        
        for setup in self.config.setups:
            results[setup] = {}
            for view_id in self.config.views:
                results[setup][view_id] = {}
                
                # Define evaluation configurations
                eval_configs = [
                    ("num_view", self.config.num_view_list, lambda x: {"num_view": x}),
                    ("num_images", self.config.num_images_list, lambda x: {"num_images": x})
                ]
                
                for config_type, config_list, param_builder in eval_configs:
                    for config_value in config_list:
                        # For num_view, only test novel views unless num_view=25
                        if config_type == "num_view" and config_value != 25 and view_id not in self.config.novel_views:
                            continue
                            
                        current_case += 1
                        case_key = f"{config_type}_{config_value}"
                        print(f"\n[{current_case}/{total_cases}] Evaluating {setup}-{view_id}-{case_key}")
                        
                        case_results = self._evaluate_single_case(setup, view_id, **param_builder(config_value))
                        results[setup][view_id][case_key] = case_results
                        
                        # Save intermediate results
                        self._save_results(results)
        
        print("\nEvaluation completed!")
        print(f"Detailed results saved to: {osp.join(self.config.output_dir, 'metrics_results.json')}")
        
        # Calculate averages across views and setups
        averaged_results = self._calculate_averages(results)
        
        # Save averaged results
        self._save_averaged_results(averaged_results)
        
        return results
    
    def _calculate_averages(self, results: Dict) -> Dict:
        """Calculate averages across all views and setups for each num_view and num_images."""
        averaged_results = {
            "num_view": {},
            "num_images": {}
        }
        
        # Define configurations to process
        config_groups = [
            ("num_view", self.config.num_view_list),
            ("num_images", self.config.num_images_list)
        ]
        
        for config_type, config_list in config_groups:
            for config_value in config_list:
                case_key = f"{config_type}_{config_value}"
                
                # Separate metrics for trained and novel views
                trained_metrics = []
                novel_metrics = []
                all_metrics = []
                
                # Collect all metrics for this configuration
                for setup in self.config.setups:
                    for view_id in self.config.views:
                        if (setup in results and 
                            view_id in results[setup] and 
                            case_key in results[setup][view_id] and
                            "sl_mask" in results[setup][view_id][case_key]):
                            
                            metrics = results[setup][view_id][case_key]["sl_mask"]
                            all_metrics.append(metrics)
                            
                            # Separate by view type
                            if view_id in self.config.trained_views:
                                trained_metrics.append(metrics)
                            elif view_id in self.config.novel_views:
                                novel_metrics.append(metrics)
                
                # Calculate averages for this configuration
                if all_metrics:
                    averaged_results[config_type][config_value] = {
                        "all_views": {
                            "average_metrics": self._compute_average_metrics(all_metrics),
                            "total_cases": len(all_metrics)
                        }
                    }
                    
                    if trained_metrics:
                        averaged_results[config_type][config_value]["trained_views"] = {
                            "average_metrics": self._compute_average_metrics(trained_metrics),
                            "total_cases": len(trained_metrics)
                        }
                    
                    if novel_metrics:
                        averaged_results[config_type][config_value]["novel_views"] = {
                            "average_metrics": self._compute_average_metrics(novel_metrics),
                            "total_cases": len(novel_metrics)
                        }
        
        return averaged_results
    
    def _compute_average_metrics(self, all_metrics: List[Dict]) -> Dict:
        """Compute average across a list of metric dictionaries."""
        if not all_metrics:
            return {}
        
        # Get all metric names from the first entry
        metric_names = all_metrics[0].keys()
        avg_metrics = {}
        
        for metric_name in metric_names:
            values = [metrics[metric_name] for metrics in all_metrics if metric_name in metrics]
            if values:
                avg_metrics[metric_name] = sum(values) / len(values)
        
        return avg_metrics
    
    def _save_results(self, results: Dict):
        """Save results to JSON file."""
        json_path = osp.join(self.config.output_dir, "metrics_results.json")
        save_metrics_to_json(results, json_path)
    
    def _save_averaged_results(self, averaged_results: Dict):
        """Save averaged results to JSON file."""
        json_path = osp.join(self.config.output_dir, "averaged_metrics_results.json")
        save_metrics_to_json(averaged_results, json_path)
        print(f"\nAveraged results saved to: {json_path}")
        
        # Also print a summary of averaged results
        print("\n" + "="*60)
        print("SUMMARY OF AVERAGED RESULTS")
        print("="*60)
        
        # Define display configurations
        display_configs = [
            ("NUM_VIEW", "num_view"),
            ("NUM_IMAGES", "num_images")
        ]
        
        for display_name, config_key in display_configs:
            if config_key in averaged_results:
                print(f"\n{display_name} Results:")
                for config_value, data in averaged_results[config_key].items():
                    print(f"\n  {config_key}={config_value}:")
                    
                    # Print all views average
                    if "all_views" in data:
                        all_data = data["all_views"]
                        print(f"    All Views (across {all_data['total_cases']} cases):")
                        for metric, value in all_data["average_metrics"].items():
                            print(f"      {metric}: {value:.4f}")
                    
                    # Print trained views average
                    if "trained_views" in data:
                        trained_data = data["trained_views"]
                        print(f"    Trained Views (across {trained_data['total_cases']} cases):")
                        for metric, value in trained_data["average_metrics"].items():
                            print(f"      {metric}: {value:.4f}")
                    
                    # Print novel views average
                    if "novel_views" in data:
                        novel_data = data["novel_views"]
                        print(f"    Novel Views (across {novel_data['total_cases']} cases):")
                        for metric, value in novel_data["average_metrics"].items():
                            print(f"      {metric}: {value:.4f}")
        
        print("\n" + "="*60)
    
    def _confirm_start(self) -> bool:
        """Ask user confirmation before starting evaluation."""
        print("Ready to start evaluation!")
        # user_input = input("Start benchmarking process? Type 'yes' to continue: ").strip().lower()
        # return user_input == 'yes'
        return True  # Auto-confirm for non-interactive environments


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Evaluation Script')
    parser.add_argument('--models_dir', type=str, required=True, help='Directory containing model outputs')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory containing ground truth data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save evaluation results')
    parser.add_argument('--extension', type=str, default='.png', help='Image file extension (default: .png)')
    parser.add_argument('--disable_depth', action='store_true', help='Disable depth evaluation')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID to use (default: 0)')
    
    # Configuration lists
    parser.add_argument('--setups', nargs='+', default=None, help='List of setups to evaluate (default: all setups)')
    parser.add_argument('--views', type=int, nargs='+', default=None, help='List of view IDs to evaluate (default: all views)')
    parser.add_argument('--num_view_list', type=int, nargs='+', default=None, help='List of num_view values to evaluate (default: [25])')
    parser.add_argument('--num_images_list', type=int, nargs='+', default=None, help='List of num_images values to evaluate (default: [])')
    
    args = parser.parse_args()
    
    # Set GPU device before creating any CUDA tensors
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    torch.cuda.set_device(0)  # After setting CUDA_VISIBLE_DEVICES, the device becomes 0
    
    config = EvaluationConfig(
        models_dir=args.models_dir,
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        extension=args.extension,
        enable_depth=not args.disable_depth,
        gpu_id=args.gpu_id,
        setups=args.setups,
        views=args.views,
        num_view_list=args.num_view_list if args.num_view_list else [25],
        num_images_list=args.num_images_list if args.num_images_list else []
    )
    evaluator = ModelEvaluator(config)
    evaluator.run_evaluation()

